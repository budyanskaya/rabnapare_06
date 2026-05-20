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

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# ============================================
# Индивидуальное задание - Вариант 2
# ============================================
variant = 2

# Параметры по таблице варианта 2
hidden_size = 32           # количество скрытых нейронов
activation_name = "Tanh"   # функция активации
optimizer_name = "Adam"    # оптимизатор
learning_rate = 0.05       # learning rate
batch_size = 32            # размер батча
epochs = 5                 # количество эпох

# Дополнительные параметры
input_size = 28 * 28       # размерность входного слоя (784 пикселя)
num_classes = 10           # 10 классов одежды
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Используемое устройство: {device}")
print(f"Параметры модели: Скрытых нейронов={hidden_size}, Активация={activation_name}, "
      f"Оптимизатор={optimizer_name}, LR={learning_rate}, Batch_size={batch_size}, Эпох={epochs}")

# ============================================
# 1. Загрузка данных FashionMNIST
# ============================================
print("\n--- Загрузка данных FashionMNIST ---")

# Преобразование: в тензор и нормализация к диапазону [0, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # преобразуем 28x28 в вектор 784
])

# Загрузка обучающей и тестовой выборок
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Обучающая выборка: {len(train_dataset)} изображений")
print(f"Тестовая выборка: {len(test_dataset)} изображений")

# Названия классов FashionMNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ============================================
# 2. Создание модели с одним скрытым слоем
# ============================================
class OneHiddenLayerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_name):
        super(OneHiddenLayerModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Выбор функции активации
        if activation_name == "Tanh":
            self.activation = nn.Tanh()
        elif activation_name == "ReLU":
            self.activation = nn.ReLU()
        elif activation_name == "Sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Неизвестная функция активации: {activation_name}")

        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# Создание модели
model = OneHiddenLayerModel(input_size, hidden_size, num_classes, activation_name)
model = model.to(device)
print(f"\n--- Модель ---")
print(model)

# Подсчёт количества параметров
total_params = sum(p.numel() for p in model.parameters())
print(f"Всего параметров: {total_params}")

# ============================================
# 3. Настройка функции потерь и оптимизатора
# ============================================
criterion = nn.CrossEntropyLoss()

# Выбор оптимизатора
if optimizer_name == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif optimizer_name == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
elif optimizer_name == "RMSprop":
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
else:
    raise ValueError(f"Неизвестный оптимизатор: {optimizer_name}")

print(f"\nОптимизатор: {optimizer}")
print(f"Функция потерь: CrossEntropyLoss")

# ============================================
# 4. Обучение модели
# ============================================
print(f"\n--- Обучение ({epochs} эпох) ---")

train_losses = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Обнуление градиентов
        optimizer.zero_grad()

        # Прямой проход
        output = model(data)
        loss = criterion(output, target)

        # Обратный проход
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Эпоха {epoch+1}/{epochs}, Средняя потеря: {avg_loss:.4f}")

# ============================================
# 5. Оценка качества на тестовой выборке
# ============================================
print("\n--- Оценка на тестовой выборке ---")

model.eval()
correct = 0
total = 0
all_preds = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

accuracy = 100 * correct / total
print(f"\n*** Итоговая точность (accuracy) на тестовой выборке: {accuracy:.2f}% ***")

# ============================================
# 6. Матрица ошибок и таблица по классам
# ============================================
print("\n--- Отчёт по каждому классу ---")

# Классификационный отчёт
report = classification_report(all_targets, all_preds, target_names=class_names)
print(report)

# Матрица ошибок
cm = confusion_matrix(all_targets, all_preds)

# Визуализация матрицы ошибок
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'Матрица ошибок (Confusion Matrix)\nВариант {variant}: hidden_size={hidden_size}, activation={activation_name}')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Таблица точности по классам
print("\n--- Таблица точности по классам ---")
class_accuracy = []
for i in range(num_classes):
    class_mask = np.array(all_targets) == i
    class_correct = np.sum((np.array(all_preds) == i) & class_mask)
    class_total = np.sum(class_mask)
    class_acc = 100 * class_correct / class_total if class_total > 0 else 0
    class_accuracy.append(class_acc)
    print(f"{class_names[i]:15s}: {class_acc:.2f}% ({class_correct}/{class_total})")

# ============================================
# 7. Выводы
# ============================================
print("\n" + "="*60)
print("ВЫВОДЫ")
print("="*60)
```

# Результаты обучения модели (Вариант 2)

**Используемое устройство:** `cuda`

---

## Параметры модели

| Параметр | Значение |
|----------|----------|
| Скрытых нейронов | 32 |
| Активация | Tanh |
| Оптимизатор | Adam |
| Learning rate | 0.05 |
| Batch size | 32 |
| Эпох | 5 |
| Всего параметров | 25 450 |

---

## Процесс обучения

| Эпоха | Средняя потеря (Loss) |
|-------|----------------------|
| 1/5   | 0.9617 |
| 2/5   | 0.8956 |
| 3/5   | 0.9039 |
| 4/5   | 0.8362 |
| 5/5   | 0.8229 |

---

## Итоговая точность (accuracy)

**68.62%**

---

## Детальный отчёт по каждому классу

| Класс | Precision | Recall | F1-score | Правильно/Всего |
|-------|-----------|--------|----------|-----------------|
| T-shirt/top | 0.75 | 0.77 | 0.76 | 770/1000 |
| Trouser | 0.97 | 0.91 | 0.94 | 912/1000 |
| Pullover | 0.76 | 0.41 | 0.53 | 407/1000 |
| Dress | 0.78 | 0.73 | 0.75 | 729/1000 |
| Coat | 0.36 | 0.92 | 0.51 | 920/1000 |
| Sandal | 0.71 | 0.75 | 0.73 | 749/1000 |
| Shirt | 0.75 | 0.00 | 0.01 | 3/1000 |
| Sneaker | 0.91 | 0.57 | 0.70 | 568/1000 |
| Bag | 0.95 | 0.89 | 0.92 | 890/1000 |
| Ankle boot | 0.67 | 0.91 | 0.78 | 914/1000 |

---

## Таблица точности по классам

| Класс | Точность |
|-------|----------|
| T-shirt/top | 77.00% (770/1000) |
| Trouser | 91.20% (912/1000) |
| Pullover | 40.70% (407/1000) |
| Dress | 72.90% (729/1000) |
| Coat | 92.00% (920/1000) |
| Sandal | 74.90% (749/1000) |
| Shirt | 0.30% (3/1000) |
| Sneaker | 56.80% (568/1000) |
| Bag | 89.00% (890/1000) |
| Ankle boot | 91.40% (914/1000) |

---

## Матрица ошибок (Confusion Matrix)

<img width="927" height="790" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/7e6ed3c2-9982-4aa4-bf8f-db278ea5cc28" />

**Наиболее частые ошибки:**
- **Shirt** ↔ **T-shirt/top** — взаимная путаница, визуально почти неотличимы
- **Pullover** ↔ **Coat** — оба предмета верхней одежды с похожей формой
- **Sneaker** ↔ **Ankle boot** — оба типа обуви с высоким верхом

---

## Оценка выбранных параметров

| Параметр | Оценка | Пояснение |
|----------|--------|-----------|
| hidden_size=32 | удовлетворительно | Минимально достаточный размер. Модель не переобучается, но может не хватать ёмкости для сложных паттернов. Увеличение до 128-256 повысило бы точность до 88-90% |
| Tanh | удовлетворительно | Работает стабильно, но ReLU обычно даёт на 2-3% выше точность из-за отсутствия проблемы "исчезающего градиента" |
| Adam | хорошо | Адаптивный оптимизатор, быстро сходится, устойчив к выбору LR |
| LR=0.05 | плохо | Слишком высок для Adam. Оптимальный диапазон: 0.001-0.01. Высокий LR мог вызвать "перескоки" минимума |
| batch_size=32 | хорошо | Стандартное значение, даёт баланс между скоростью обучения и стабильностью градиента |
| epochs=5 | недостаточно | Мало для полной сходимости. Потери продолжали снижаться к 5-й эпохе. Нужно минимум 15-20 эпох |
| 1 скрытый слой | удовлетворительно | Для FashionMNIST достаточно, но добавление второго слоя повысило бы качество на сложных классах |

---

## Что дало хороший результат

- Использование **Adam** вместо SGD — быстрая и стабильная сходимость
- **Нормализация данных** (преобразование в тензор)
- Один скрытый слой достаточен для FashionMNIST (не очень сложная задача)

---

## Что дало плохой результат

- **Слишком высокий learning rate (0.05)** — возможны "перескоки" минимума
- **Малое количество эпох (5)** — потеря около 3-5% точности
- **Tanh** — уступает ReLU (ReLU обычно даёт на 2-3% выше точность)

---

## Рекомендации по улучшению

1. Увеличить количество эпох до **15-20**
2. Уменьшить learning rate до **0.001** (для Adam)
3. Использовать **ReLU** вместо Tanh
4. Добавить **Dropout** для регуляризации
5. Увеличить **hidden_size** до 128-256
6. Использовать **пакетную нормализацию** (BatchNorm)

---

## Вывод

Модель с одним скрытым слоем (32 нейрона) и активацией Tanh показала **удовлетворительное качество (68.62%)**, но далека от оптимального. Основные проблемы:

-  **Слишком мало эпох (5)** — модель не успела сойтись
-  **Слишком высокий LR (0.05)** — возможны колебания на поздних эпохах
-  **Tanh** — работает, но ReLU эффективнее

**Ожидаемая максимальная точность** для данной архитектуры при оптимальных параметрах составляет **~88-90%**. Текущая точность ниже преимущественно из-за малого количества эпох и завышенного learning rate.
