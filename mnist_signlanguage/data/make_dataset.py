import pandas as pd
import torch


def process_and_save_data(csv_file_path, labels_file_path, images_tensor_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Extract labels and image data
    labels = df['label'].values
    images = df.drop('label', axis=1).values
    images = images.reshape(-1, 28, 28)  # Reshape to 28x28 images
    images_tensor = torch.tensor(images, dtype=torch.float32)

    # Save labels to a text file
    with open(labels_file_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

    # Save images as tensors in a .pt file
    torch.save(images_tensor, images_tensor_file_path)

def fetch_dataloader(cfg):
    DATADIR = cfg.dir
    BATCH_SIZE = cfg.batch_size



    img_path_train = 'C:/Programmering/mlo_project_indiv/project1/data/corruptmnist/train_images_0.pt' 
    label_path_train = 'C:/Programmering/mlo_project_indiv/project1/data/corruptmnist/train_target_0.pt'
    img_path_test = 'C:/Programmering/mlo_project_indiv/project1/data/corruptmnist/test_images.pt'
    label_path_test = 'C:/Programmering/mlo_project_indiv/project1/data/corruptmnist/test_target.pt'

    data_train = torch.load(imgPathTrain)
    labels_train = torch.load(labelPathTrain)
    data_test = torch.load(imgPathTest)
    labels_test = torch.load(labelPathTest)
    

    train_tataset = [[x, y] for x, y in zip(dataTrain, labelsTrain)]
    test_Tataset = [[x, y] for x, y in zip(dataTest, labelsTest)]


    train = DataLoader(trainDataset, batch_size=64, shuffle=True)
    test = DataLoader(testDataset, batch_size=64, shuffle=True)


    return train, test

if __name__ == '__main__':
    # File paths for the training and test datasets
    # Update with your train CSV file path
    train_file_path = 'data/raw/sign_mnist_train/sign_mnist_train.csv'
    # Update with your test CSV file path
    test_file_path = 'data/raw/sign_mnist_test/sign_mnist_test.csv'

    # File paths for output for training data
    # Update with output path for train labels
    train_labels_file = 'data/processed/train/train_labels.txt'
    # Update with output path for train image tensors
    train_images_tensor_file = 'data/processed/train/train_images.pt'

    # Process and save training data
    process_and_save_data(
        train_file_path, train_labels_file, train_images_tensor_file)

    # File paths for output for test data
    # Update with output path for test labels
    test_labels_file = 'data/processed/test/test_labels.txt'
    # Update with output path for test image tensors
    test_images_tensor_file = 'data/processed/test/test_images.pt'

    # Process and save test data
    process_and_save_data(test_file_path, test_labels_file,
                          test_images_tensor_file)
    pass
