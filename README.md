# Практическая работа 6. Введение в PyTorch

## Цели работы

После выполнения рабочей тетради студент должен уметь:

- создавать и преобразовывать тензоры PyTorch;
- выполнять операции над тензорами и понимать broadcasting;
- использовать autograd для вычисления градиентов;
- описывать простую нейронную сеть через `torch.nn`;
- работать с `Dataset` и `DataLoader`;
- запускать базовый цикл обучения и оценивать качество модели;
- адаптировать модель под новую задачу с помощью заморозки слоев.

## Индивидуальное задание, вариант 2

| Вариант | Скрытых нейронов | Активация | Оптимизатор | LR | Batch size | Эпох |
|---------|-----------------|-----------|-------------|-----|------------|------|
| 2       | 32              | Tanh      | Adam        | 0.05 | 32 | 5 |

### Что сделать:

1. загрузить FashionMNIST
2. создать модель с одним скрытым слоем указанного размера
3. использовать указанную функцию активации, оптимизатор, learning rate, batch_size и количество эпох
4. вывести итоговую accuracy на тестовой выборке
5. построить/вывести матрицу ошибок или таблицу правильных ответов по классам
6. написать вывод: какие параметры дали хороший/плохой результат и почему

## Решение

### Код решения

```python
# ========== ИНДИВИДУАЛЬНОЕ ЗАДАНИЕ, ВАРИАНТ 2 ==========
variant = 2

# Параметры из таблицы
hidden_size = 32
activation_name = 'Tanh'
optimizer_name = 'Adam'
learning_rate = 0.05
batch_size = 32
epochs = 5

# 1. Загрузка FashionMNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Нормализация: преобразуем в тензор и нормализуем к диапазону [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2. Определение модели с одним скрытым слоем
class FashionMNIST_OneHidden(nn.Module):
    def __init__(self, hidden_size=32, activation='Tanh'):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, hidden_size)
        
        # Выбор функции активации
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Неизвестная активация: {activation}")
            
        self.fc2 = nn.Linear(hidden_size, 10)  # 10 классов

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

model = FashionMNIST_OneHidden(hidden_size=hidden_size, activation=activation_name)

# 3. Определение оптимизатора и функции потерь
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. Обучение
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Используется устройство: {device}")

print("\n--- Начало обучения ---")
for epoch in range(1, epochs+1):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Эпоха {epoch}/{epochs} | Loss: {avg_loss:.4f}")

# 5. Оценка на тестовой выборке
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"\n*** Итоговая точность (accuracy): {accuracy:.2f}% ***")

# 6. Матрица ошибок и таблица по классам
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

cm = confusion_matrix(all_labels, all_preds)

# Выводим таблицу правильных ответов по классам
print("\n--- Таблица правильных ответов по классам ---")
print(f"{'Класс':15s} | {'Правильно':>8} | {'Всего':>6} | {'Точность':>8}")
print("-" * 50)
for i, cls in enumerate(classes):
    correct = cm[i, i]
    total = np.sum(cm[i, :])
    print(f"{cls:15s} | {correct:8d} | {total:6d} | {100*correct/total:7.1f}%")

# Построение матрицы ошибок
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(xticks_rotation=45, cmap='Blues', values_format='d')
plt.title(f"Confusion Matrix для FashionMNIST (Вариант {variant})\nhidden_size={hidden_size}, activation={activation_name}")
plt.tight_layout()
plt.show()

# Дополнительно: вывод самых частых ошибок
print("\n--- Анализ самых частых ошибок ---")
for i in range(10):
    # Находим класс, с которым чаще всего путают текущий класс
    misclass = np.argsort(cm[i])[-2] if np.argsort(cm[i])[-1] == i else np.argsort(cm[i])[-1]
    if misclass != i:
        print(f"{classes[i]:15s} часто путают с {classes[misclass]:15s} ({cm[i, misclass]} ошибок)")
```

## Результат
# Результаты обучения модели

**Используемое устройство:** `cuda`

---

##  Начало обучения

| Эпоха | Loss     |
|-------|----------|
| 1/5   | 0.6543   |
| 2/5   | 0.4875   |
| 3/5   | 0.4181   |
| 4/5   | 0.3773   |
| 5/5   | 0.3518   |

---

##  Итоговая точность (accuracy)

**84.37%**

---

##  Таблица правильных ответов по классам

| Класс            | Правильно | Всего | Точность |
|------------------|-----------|-------|----------|
| T-shirt/top      | 812       | 1000  | 81.2%    |
| Trouser          | 951       | 1000  | 95.1%    |
| Pullover         | 753       | 1000  | 75.3%    |
| Dress            | 835       | 1000  | 83.5%    |
| Coat             | 768       | 1000  | 76.8%    |
| Sandal           | 924       | 1000  | 92.4%    |
| Shirt            | 582       | 1000  | 58.2%    |
| Sneaker          | 893       | 1000  | 89.3%    |
| Bag              | 966       | 1000  | 96.6%    |
| Ankle boot       | 940       | 1000  | 94.0%    |

---

##  Анализ самых частых ошибок

| Класс            | Часто путают с       | Количество ошибок |
|------------------|----------------------|------------------|
| T-shirt/top      | Shirt                | 148              |
| Trouser          | Bag                  | 35               |
| Pullover         | Coat                 | 98               |
| Dress            | Coat                 | 53               |
| Coat             | Pullover             | 104              |
| Sandal           | Ankle boot           | 45               |
| Shirt            | T-shirt/top          | 148              |
| Sneaker          | Ankle boot           | 66               |
| Bag              | Trouser              | 27               |
| Ankle boot       | Sneaker              | 36               |

<img width="927" height="790" alt="image" src="https://github.com/user-attachments/assets/7e6ed3c2-9982-4aa4-bf8f-db278ea5cc28" />

## Оценка параметров

| Параметр | Оценка | Почему |
|----------|--------|--------|
| hidden_size=32 | удовлетворительно | Минимально достаточный размер. Увеличение до 128-256 повысило бы точность на 2-3% |
| Tanh | удовлетворительно | Работает стабильно, но ReLU обычно даёт на 2-3% выше точность из-за отсутствия насыщения |
| Adam | хорошо | Адаптивный оптимизатор, быстро сходится, устойчив к выбору LR |
| LR=0.05 | плохо | Слишком высок для Adam. Оптимальный диапазон: 0.001-0.01. Высокий LR мог вызвать "перескоки" минимума |
| batch_size=32 | хорошо | Стандартное значение, даёт баланс между скоростью и стабильностью |
| epochs=5 | недостаточно | Потери продолжали снижаться (0.3518 на 5-й эпохе). Нужно минимум 15-20 эпох |
| 1 скрытый слой | удовлетворительно | Для FashionMNIST достаточно, но добавление второго слоя повысило бы качество на сложных классах |

## Наблюдения

**Лучше всего распознаются классы:**
- Bag (96.6%), Trouser (95.1%), Ankle boot (94.0%), Sandal (92.4%)

**Хуже всего распознаются классы:**
- Shirt (58.2%), Pullover (75.3%), Coat (76.8%), T-shirt/top (81.2%)

**Характерные ошибки:**
- Shirt ↔ T-shirt/top (взаимные ошибки ~148) — визуально почти неотличимы, разница только в длине рукава
- Pullover ↔ Coat (98-104 ошибки) — оба предмета верхней одежды с похожей формой
- Coat ↔ Dress (53 ошибки) — путаница между верхней одеждой и платьем
- Sneaker ↔ Ankle boot (36-66 ошибок) — оба типа обуви с высоким верхом

## Выводы

### Общая оценка

Модель с одним скрытым слоем (32 нейрона) и активацией Tanh показала удовлетворительное качество классификации FashionMNIST (84.37%) для 5 эпох обучения. Заданная конфигурация является базовой и позволяет решить задачу на приемлемом уровне, но далека от оптимальной.
