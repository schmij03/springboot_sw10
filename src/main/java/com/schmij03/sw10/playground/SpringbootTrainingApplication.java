package com.schmij03.sw10.playground;

import java.io.IOException;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;

@SpringBootApplication
public class SpringbootTrainingApplication {

    public SpringbootTrainingApplication() throws TranslateException, IOException {

    }

    public static void main(String[] args) throws TranslateException, IOException {
        SpringApplication.run(SpringbootTrainingApplication.class, args);

        // Create an instance of the Models class
        Models models = new Models();
        final String MODEL_NAME = "shoeclassifier";

        // Initialize the image dataset
        ImageFolder dataset = models.initDataset("C:/Users/jansc/Desktop/MODEL_DEPLOYMENT/sw9_training/Springboot_SW9/playground/src/main/resources/static/data/ut-zap50k-images-square");
        RandomAccessDataset[] datasets = dataset.randomSplit(8, 2);

        RandomAccessDataset trainingSet = datasets[0];
        RandomAccessDataset testSet = datasets[1];

        // Get the model for shoe classification
        models.getModel(MODEL_NAME);

        // Configure the training loss
        Loss loss = Loss.softmaxCrossEntropyLoss();

        // Setup the training configuration
        TrainingConfig config = models.setupTrainingConfig(loss);

        // Train the model
        models.train(trainingSet, testSet, config, MODEL_NAME);
    }
}