import matplotlib.pyplot as plt
from torchvision import transforms
from model import *
from dataloader import *
import torch




hyper_param_epoch = 500
hyper_param_batch = 256
hyper_param_learning_rate = 0.0001

transforms_train = transforms.Compose([transforms.Resize((256, 256)),transforms.RandomRotation(10.),transforms.ToTensor()])
transforms_test = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])


train_data_set = CustomImageDataset(data_set_path="/home/nripendra/Multiple_treatment&Diagnosis/Experiment/oriented_detection/TND_classification/train", transforms=transforms_train)

print("Total  train data  : " , len(train_data_set))
train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

test_data_set = CustomImageDataset(data_set_path="/home/nripendra/Multiple_treatment&Diagnosis/Experiment/oriented_detection/TND_classification/test_classifier_image", transforms=transforms_test)
print("Total  Test  data  : " , len(test_data_set))
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)

if not (train_data_set.num_classes == test_data_set.num_classes):
    print("error: Numbers of class in training set and test set are not equal")
    exit()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = train_data_set.num_classes
labels = test_data_set.labels
custom_model = CustomConvNet(num_classes=num_classes).to(device)
#custom_model.load_state_dict(torch.load("/home/nripendra/Multiple_treatment&Diagnosis/Experiment/oriented_detection/TND_classification/model.pth", map_location=device))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)

epoch_list  = []
losses_list  = []
acc_list = []


def get_model_acc(custom_model):
    custom_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for item in test_loader:
            images = item['image'].to(device)
            labels = item['label'].to(device)
            outputs = custom_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
        print(correct/total)
        return (correct / total)

for e in range(hyper_param_epoch):
    temp_loss = 0
    custom_model.train()
    for i_batch, item in enumerate(train_loader):
        images = item['image'].to(device)
        labels = item['label'].to(device)
        # Forward pass
        outputs = custom_model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i_batch + 1) % 1  == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(e + 1, hyper_param_epoch, loss.item()))
            temp_loss = loss.item()
    acc = get_model_acc(custom_model)
    epoch_list.append(e)
    losses_list.append(temp_loss)
    acc_list.append(acc)
    plt.plot(epoch_list, losses_list, label="Train label ")
    plt.title("Training Cruve  between epoch and losss")
    plt.ylabel("Loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig("image.png")

# Test the model

#
#
# plt.plot(epoch_list , losses_list , label  = "Train label ")
# plt.plot(epoch_list , acc_list , label  = "Model ACC  ")
# plt.title("Training Cruve  between Epoch VS Loss VS Accuracy ")
# plt.ylabel("Loss and Accuracy ")
# plt.xlabel("Epochs")
# plt.legend()
# plt.savefig("train_and_acc.png")
# plt.clf()
#
#
#
# plt.plot(epoch_list , losses_list , label  = "Loss")
# plt.title("Training Loss vs Epoch")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.savefig("image_loss.png")
# plt.clf()
#
#
# plt.plot(epoch_list , acc_list , label  = "Model ACC  ")
# plt.title("Training Accuracy vs Epoch ")
# plt.ylabel("Accuracy")
# plt.xlabel("Epochs")
# plt.legend()
# plt.savefig("image_acc.png")
# plt.clf()


custom_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for item in test_loader:
        images = item['image'].to(device)
        labels = item['label'].to(device)
        outputs = custom_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))


PATH = "model_new.pth"
torch.save(custom_model.state_dict(), PATH)