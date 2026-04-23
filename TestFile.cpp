#include "Matrix.h"
#include "Node.h"
#include "MLP.h"
#include "Layer.h"
#include <cmath>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <string>
#include <queue>

// See XOR problem in action
void XOR()
{
    double learningRate = 0.1;
    MLP mlp(2, {2, 1}, learningRate);

    Matrix X_train(4, 2, {0, 0, 0, 1, 1, 0, 1, 1});
    Matrix y_train(4, 1, {0, 1, 1, 0});

    std::cout << "Network architecture -> Input " << X_train.getRows() << "x" << X_train.getCols() << " ";
    std::cout << "\n";

    std::this_thread::sleep_for(std::chrono::seconds(2));

    Nodeptr input = Node::create(X_train);
    Nodeptr y_true = Node::create(y_train);

    const int epochs = 16000;
    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        mlp.zeroGrad();

        // Forward pass
        auto output = mlp.forward(input);

        // Calculate loss
        auto loss = Node::bce(y_true, output);
        // loss->getData().toString();

        // Backward pass
        loss->backprop();

        // Update weights
        mlp.update(learningRate);

        // Print progress
        if (epoch % 100 == 0)
        {
            auto preds = output->getData();
            double correct = 0;

            // std::cout << (preds.getRows()) << std::endl;
            for (size_t i = 0; i < preds.getRows(); i++)
            {
                // std::cout << preds.get(i, 0) << std::endl;
                double pred = preds.get(i, 0) > 0.5 ? 1.0 : 0.0;
                if (pred == y_train.get(i, 0))
                    correct++;
            }

            std::cout << "Epoch " << epoch
                      << " | Loss: " << loss->getData().get(0, 0)
                      << " | Accuracy: " << (correct / 4.0) * 100 << "%\n";
        }
    }
}

// Helper func used in MNIST
std::unordered_map<int, Nodeptr> read_last_n_lines(std::ifstream &fs, size_t n)
{
    fs.seekg(0, std::ios_base::end);       // File pointer seeks to end of file (relative)
    std::streampos file_size = fs.tellg(); /** File pointer is on last byte of file, so get file_size
                                               Note, this is possible since tellg returns
                                               absolute positions from the beginning */

    size_t count = 0;
    std::streamoff pos = file_size - static_cast<std::streamoff>(1); // Calculate current position (go to last character)

    while (pos >= 0 && count < n) // Continue while we haven't gotten the last 10 lines
    {
        fs.seekg(pos, std::ios_base::beg); // Find the absolute position from the beginning, value 'pos'
        if (fs.peek() == '\n')             // Look at current character in fil
        {
            count++;
        }

        if (count == n)
        {
            pos += 1;
            fs.seekg(pos, std::ios_base::beg); // Go back one pos, as we've reached 10 lines already
        }
        pos -= 1; // Move position back one character
    }

    std::unordered_map<int, Nodeptr> small_test;

    std::string line;
    while (getline(fs, line))
    {
        Matrix test(1, 784); // Create matrix to hold pixel values

        std::stringstream ss(line);

        std::string num_test;

        int label_test; // Holds label of image

        size_t j = 0;        // Solely used to tell if we're at the label or not (first number)
        int pixel_count = 0; // Used to count number of pixels counted in the image

        while (getline(ss, num_test, ','))
        {

            if (j == 0) // Set label
            {
                label_test = std::stoi(num_test);
                j++;
                continue;
            }
            test.set(0, pixel_count, std::stod(num_test) / 255); // Put normalized pixel value in matrix
            pixel_count++;
            j++;
        }
        small_test[label_test] = Node::create(test); // Once done, put label matrix pair in map
    }

    return small_test;
}

// See MNIST problem in action
void MNIST()
{

    std::cout << "Please enter the expected batch size 1 - 60000 \n"
                 "~~Computer performance may vary with higher batch sizes~~ \n"
                 "Enter: ";
    size_t batches;
    while (!(std::cin >> batches) || batches <= 0)
    {
        std::cout << "Please enter a valid range 1 - 60000" << std::endl;
        std::cout << "Enter: ";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    std::cout << "\nPlease enter the expected epochs (> than 0)\n"
                 "Enter: ";
    int epochs;
    while (!(std::cin >> epochs) || epochs <= 0)
    {
        std::cout << "Please enter a valid range (> than 0)" << std::endl;
        std::cout << "Enter: ";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    std::cout
        << "\nPreprocessing data..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::ifstream training_data("mnist_train.csv");
    std::ifstream testing_data("mnist_test.csv");

    if (!training_data.is_open() || !testing_data.is_open())
    {
        std::cout << "Something went wrong" << std::endl;
        return;
    }

    std::string line;

    Matrix x_train(batches, 784);

    std::vector<int> labels_train;

    size_t i = 0;
    // Process data from MNIST training set csv into x_train
    while (getline(training_data, line))
    {
        if (i == batches)
            break;

        std::stringstream ss(line);

        std::string num_train;
        size_t j = 0;
        int pixel_count = 0;
        int label_train;

        while (getline(ss, num_train, ','))
        {
            if (j == 0) // Get label and push to our vector
            {
                label_train = std::stoi(num_train);
                labels_train.push_back(label_train);
                j++;
                continue;
            }
            x_train.set(i, pixel_count, std::stod(num_train) / 255); // normalize and add, pixel values are between 0 and 1
            pixel_count++;
            j++;
        }
        i++;
    }

    size_t max_test_case = 9990;
    Matrix x_test((batches > max_test_case) ? max_test_case : batches, 784);
    std::vector<int> labels_test;

    std::string line2;

    i = 0;
    // Process data from MNIST testing set csv into x_test
    while (getline(testing_data, line2))
    {
        if (i >= batches || i >= max_test_case)
        {
            break;
        }

        std::stringstream ss2(line2);

        std::string num_test;
        size_t j = 0;
        int pixel_count = 0;
        int label_test;

        while (getline(ss2, num_test, ','))
        {
            if (j == 0) // Get label and push to our vector
            {
                label_test = std::stoi(num_test);
                labels_test.push_back(label_test);
                j++;
                continue;
            }
            x_test.set(i, pixel_count, std::stod(num_test) / 255);
            pixel_count++;
            j++;
        }
        i++;
    }

    Matrix y_train = Matrix::one_hot_encode(labels_train, 10);
    Matrix y_test = Matrix::one_hot_encode(labels_test, 10);
    std::this_thread::sleep_for(std::chrono::seconds(2));

    double learningRate = 0.1;

    Nodeptr input_train = Node::create(x_train);
    Nodeptr input_test = Node::create(x_test);

    Nodeptr y_true_train = Node::create(y_train);
    Nodeptr y_true_test = Node::create(y_test);

    MLP mlp(784, {64, 32, 10}, learningRate);

    std::cout << "\nTraining data..."
              << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        if (epoch % 50 != 0)
        {
            std::cout << "/" << std::flush;
        }
        else
        {
            std::cout << "\n";
        }

        mlp.zeroGrad();                                  // Zero out gradients to stop accumulation
        auto training_output = mlp.forward(input_train); // Forward pass

        Nodeptr loss = Node::softmaxCE(y_true_train, training_output); // Obtain loss

        mlp.backward(loss); // Calculate gradients of each layers weights and biases in MLP

        mlp.update(learningRate); // Adjust parameters of each layer

        auto test_output = mlp.forward(input_test); // Test on test set

        // Print progress
        if (epoch % 50 == 0)
        {
            std::cout << "\nEpoch " << epoch << std::endl;

            auto preds = test_output->getData();
            double correct = 0;

            for (size_t i = 0; i < preds.getRows(); i++)
            {
                int pred_idx = 0;
                double max_prob = preds.get(i, 0);
                for (size_t j = 0; j < preds.getCols(); j++)
                {
                    if (preds.get(i, j) > max_prob)
                    {
                        max_prob = preds.get(i, j);
                        pred_idx = j;
                    }
                }

                if (pred_idx == labels_test[i])
                {
                    correct++;
                }
            }

            std::cout << "Epoch --->" << epoch
                      << " | Loss: " << loss->getData().get(0, 0)
                      << " | Test Accuracy: " << (correct / batches) * 100 << "%\n";
        }
    }

    // Optional game below

    std::cout << "------------------------------------------" << std::endl;
    std::cout << "\n~~~Fun prediction game~~~\n"
              << std::endl;
    std::cout << "Enter a number 0 - 9, see if the model is able to predict it!" << std::endl;
    std::cout << "Press -1 to break" << std::endl;

    std::cout << "Note: It has not been trained on these 10 images" << std::endl;
    int num;
    std::cout << "Enter: ";
    while (!(std::cin >> num) || num < 0 || num >= 10)
    {
        std::cout << "Please enter a valid range 0 - 9" << std::endl;
        std::cout << "Enter: ";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    std::unordered_map<int, Nodeptr> values = read_last_n_lines(testing_data, 10);

    while (num != -1)
    {
        auto input = values[num];
        auto output = mlp.forward(input);

        auto preds = output->getData();

        std::this_thread::sleep_for(std::chrono::seconds(1));

        std::cout << "\n------------------------------------------" << std::endl;
        std::cout << "MODEL OUTPUT: " << std::endl;
        preds.toString();

        int pred_idx = 0;
        double max_prob = preds.get(0, 0);
        for (size_t j = 0; j < preds.getCols(); j++)
        {
            if (preds.get(0, j) > max_prob)
            {
                max_prob = preds.get(0, j);
                pred_idx = j;
            }
        }

        std::cout << "------------------------------------------" << std::endl;
        std::cout << "Model guessed: " << pred_idx << std::endl;
        if (pred_idx == num)
        {
            std::cout << "Successfully predicted!" << std::endl;
        }
        else
        {
            std::cout << "Incorrect prediction :(" << std::endl;
        }

        std::cout << "------------------------------------------" << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(1));

        std::cout << "Enter a number 0 - 9, see if the model is able to predict it!" << std::endl;
        std::cout << "\nPress -1 to break" << std::endl;
        std::cout << "Enter: ";

        while (!(std::cin >> num) || num < 0 || num >= 10)
        {
            std::cout << "Please enter a valid range 0 - 9" << std::endl;
            std::cout << "Enter: ";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }
}

// Or trying solving your own problem
int main()
{
    MNIST();
    // XOR();
}