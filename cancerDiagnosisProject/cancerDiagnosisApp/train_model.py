import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from PIL import Image
import pandas as pd
import os
import shutil

def get_device():
    """
    Cihazı belirler. Eğer GPU varsa "cuda", yoksa "cpu" olarak döner.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def get_transform(is_training=True):
    """
    Görüntülerin ön işleme adımları:
    - Eğitim sırasında veri artırma (data augmentation)
    - Test/tahmin sırasında sadece temel dönüşümler
    """
    if is_training:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform

def organize_dataset(base_dir, metadata_file):
    """
    HAM10000 veri setini sınıflara göre organize eder
    """
    print(f"\nVeri seti organizasyonu başlıyor...")
    print(f"Metadata dosyası okunuyor: {metadata_file}")
    
    df = pd.read_csv(metadata_file)
    print(f"Toplam görsel sayısı: {len(df)}")
    
    # Sınıf dağılımını göster
    print("\nSınıf dağılımı:")
    for dx in df['dx'].value_counts().items():
        print(f"- {dx[0]}: {dx[1]} görsel")
    
    organized_dir = os.path.join(base_dir, "organized_dataset")
    print(f"\nHedef klasör: {organized_dir}")
    
    if os.path.exists(organized_dir):
        print("Organized dataset klasörü siliniyor...")
        shutil.rmtree(organized_dir)
    
    os.makedirs(organized_dir, exist_ok=True)
    print("\nHastalık sınıfları için klasörler oluşturuluyor...")
    
    # Önce tüm klasörleri oluştur
    for dx in df['dx'].unique():
        dx_dir = os.path.join(organized_dir, dx)
        os.makedirs(dx_dir, exist_ok=True)
        print(f"- {dx} klasörü oluşturuldu")
    
    print("\nGörsel klasörleri kontrol ediliyor:")
    image_dirs = [
        os.path.join(base_dir, "HAM10000_images_part_1"),
        os.path.join(base_dir, "HAM10000_images_part_2")
    ]
    
    for img_dir in image_dirs:
        if os.path.exists(img_dir):
            print(f"- {img_dir} bulundu")
            print(f"  İçindeki dosya sayısı: {len(os.listdir(img_dir))}")
        else:
            print(f"- UYARI: {img_dir} bulunamadı!")
    
    print("\nGörseller sınıflarına göre kopyalanıyor...")
    total_images = len(df)
    images_not_found = 0
    class_counts = {dx: 0 for dx in df['dx'].unique()}
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"İşlenen görsel: {idx}/{total_images}")
            
        image_id = row['image_id']
        dx = row['dx']
        image_found = False
        
        # Tüm olası görsel dosya uzantılarını dene
        for img_dir in image_dirs:
            # Önce uzantısız dosya adıyla ara
            potential_files = [f for f in os.listdir(img_dir) if f.startswith(image_id)]
            if potential_files:
                source_path = os.path.join(img_dir, potential_files[0])
                dest_path = os.path.join(organized_dir, dx, potential_files[0])
                shutil.copy2(source_path, dest_path)
                image_found = True
                class_counts[dx] += 1
                break
        
        if not image_found:
            images_not_found += 1
            if images_not_found <= 10:
                print(f"UYARI: {image_id} için görsel bulunamadı!")
            elif images_not_found == 11:
                print("Daha fazla bulunamayan görsel var...")
    
    print("\nVeri seti organizasyonu tamamlandı!")
    print(f"Toplam işlenen görsel: {total_images}")
    print(f"Bulunamayan görsel sayısı: {images_not_found}")
    
    print("\nSınıflara göre kopyalanan görsel sayıları:")
    for dx, count in class_counts.items():
        dx_dir = os.path.join(organized_dir, dx)
        actual_count = len(os.listdir(dx_dir))
        print(f"- {dx}: {count} görsel kopyalandı, klasörde {actual_count} görsel var")
    
    print(f"\nOrganized dataset klasörü: {organized_dir}")
    return organized_dir

def dataloader(base_dir, metadata_file, transform, batch_size=32, val_split=0.2):
    """
    HAM10000 veri setini yükler ve train/validation olarak böler
    """
    organized_dir = organize_dataset(base_dir, metadata_file)
    dataset = datasets.ImageFolder(root=organized_dir, transform=transform)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return dataset.classes, train_loader, val_loader

def get_class_names():
    """
    HAM10000 sınıf isimleri
    """
    return [
        'akiec',  # Actinic Keratoses and Intraepithelial Carcinoma
        'bcc',    # Basal Cell Carcinoma
        'bkl',    # Benign Keratosis-like Lesions
        'df',     # Dermatofibroma
        'mel',    # Melanoma
        'nv',     # Melanocytic Nevi
        'vasc'    # Vascular Lesions
    ]

def create_model(device, num_classes):
    """
    ResNet18 modelini oluşturur ve transfer learning için hazırlar
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    
    # Tüm katmanları donduralım
    for param in model.parameters():
        param.requires_grad = False
    
    # Son birkaç katmanı eğitim için açalım
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Sınıflandırma katmanını değiştirelim
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    
    return model.to(device)

def train_model(base_dir, metadata_file):
    """
    Modeli HAM10000 veri seti üzerinde eğitir
    """
    num_epochs = 30
    batch_size = 32
    device = get_device()
    transform = get_transform(is_training=True)
    
    # Veri setini yükle
    class_names, train_loader, val_loader = dataloader(
        base_dir=base_dir,
        metadata_file=metadata_file,
        transform=transform,
        batch_size=batch_size
    )
    
    model = create_model(device=device, num_classes=len(class_names))
    
    # Kayıp fonksiyonu ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.fc.parameters()},
        {'params': model.layer4.parameters(), 'lr': 0.0001}
    ], lr=0.001)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # Model kaydetme yolunu düzelt
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\nModel checkpointleri şu klasöre kaydedilecek: {checkpoint_dir}")
    
    best_val_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    for epoch in range(num_epochs):
        # Eğitim
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                "Loss": f"{running_loss/len(train_loader):.4f}",
                "Acc": f"{100*correct/total:.2f}%"
            })
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {100*correct/total:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate güncelleme
        scheduler.step(val_loss)
        
        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': running_loss/len(train_loader),
                'train_acc': 100*correct/total,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'class_names': class_names
            }, best_model_path)
            print(f"\nYeni en iyi model kaydedildi! Validation Accuracy: {val_acc:.2f}%")
        
        # Her epoch sonunda checkpoint kaydet
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': running_loss/len(train_loader),
            'train_acc': 100*correct/total,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'class_names': class_names
        }, checkpoint_path)
    
    print("\nEğitim tamamlandı!")
    print(f"En iyi validation accuracy: {best_val_acc:.2f}%")
    print(f"En iyi model şuraya kaydedildi: {best_model_path}")
    
    return model, class_names

def evaluate_model(model, dataloader, criterion, device):
    """
    Modeli değerlendirir
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss/len(dataloader), 100*correct/total

def predict_image(image_path):
    """
    Kaydedilmiş modeli kullanarak tahmin yapar
    """
    device = get_device()
    
    # Model dosyasının yolunu belirle
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'best_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = create_model(device=device, num_classes=len(checkpoint['class_names']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = get_transform(is_training=False)
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        result = checkpoint['class_names'][predicted.item()]
        confidence = confidence.item() * 100
        
        diagnosis_map = {
            'akiec': 'Actinic Keratoses ve Intraepithelial Carcinoma',
            'bcc': 'Basal Cell Carcinoma',
            'bkl': 'Benign Keratosis Benzeri Lezyonlar',
            'df': 'Dermatofibroma',
            'mel': 'Melanoma',
            'nv': 'Melanocytic Nevi',
            'vasc': 'Vasküler Lezyonlar'
        }
        
        diagnosis = diagnosis_map.get(result, result)
        
        return f"{diagnosis} (Güven: {confidence:.1f}%)"

def convert_excel_to_csv(excel_path, save_path=None):
    """
    Excel dosyasını CSV'ye dönüştürür
    """
    if save_path is None:
        save_path = os.path.splitext(excel_path)[0] + '.csv'
    
    df = pd.read_excel(excel_path)
    df.to_csv(save_path, index=False)
    print(f"Excel dosyası CSV'ye dönüştürüldü: {save_path}")
    return save_path




















    
