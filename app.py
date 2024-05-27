import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Veriyi oku ve 'sorgu' ve 'label' sütunlarını al
df = pd.read_csv('banka.csv')
df = df[['sorgu', 'label']]

# Stopwords listesini tanımla
stopwords = [
    'fakat', 'lakin', 'ancak', 'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 
    'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 
    'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 
    'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 
    'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 
    'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani'
]

# Kullanıcıdan yeni bir mesaj al
mesaj = input("Yapmak istediğiniz işlemi giriniz: ")

# Yeni mesajı DataFrame'e ekle, 'label' değerini geçici olarak 'yeni_sınıf' olarak ayarla
mesajdf = pd.DataFrame({'sorgu': [mesaj], 'label': ['yeni_sınıf']})
df = pd.concat([df, mesajdf], ignore_index=True)

# Stopwords'leri kaldır
for word in stopwords:
    df['sorgu'] = df['sorgu'].str.replace(r'\b' + word + r'\b', '', regex=True)

# Büyük harfleri küçük harfe çevir
df['sorgu'] = df['sorgu'].str.lower()

# CountVectorizer'ı tanımla, max_features=200 parametresiyle en sık kullanılan 200 kelimeyi al
cv = CountVectorizer(max_features=200)
x = cv.fit_transform(df['sorgu']).toarray()

# Etiketleri sayısal değerle kodlayalım
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'].astype(str))
y = df['label'].values

# Veriyi tensorlere dönüştür
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Eğitim ve test veri kümelerine ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.8)

# Yapay Sinir Ağı Modelini Oluştur
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)  # Dropout ekle

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Gerekli boyutlar
input_size = x.shape[1]
hidden_size = 256  # Daha geniş bir gizli katman boyutu
output_size = len(le.classes_)  # Geçici sınıfı ekle

model = Net(input_size, hidden_size, output_size)

# Loss ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Modeli Eğit
epochs = 50  # Eğitim süresini artır
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Modelin Performansını Değerlendir
model.eval()
with torch.no_grad():
    outputs = model(x_test)
    _, predicted = torch.max(outputs, 1)
    print(classification_report(y_test, predicted, labels=range(len(le.classes_)), target_names=le.classes_))


# Yeni mesajın tahminini al
y_pred_new_message = model(x[-1].unsqueeze(0))
_, predicted_class = torch.max(y_pred_new_message, 1)
predicted_class = predicted_class.item()

# Tahmin edilen sınıfı yazdır
predicted_label = le.inverse_transform([predicted_class])[0]
print(f'\nSonuç: {predicted_label}')

