package com.schmij03.sw10.playground;

import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.*;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Paths;

public class Models {
    // Initialize the image dataset
    public static ImageFolder initDataset(String datasetRoot)
            throws IOException, TranslateException {

        ImageFolder dataset = ImageFolder.builder()
                .setRepositoryPath(Paths.get(datasetRoot))
                .optMaxDepth(10)
                .addTransform(new Resize(224, 224))
                .addTransform(new ToTensor())
                .setSampling(32, true)
                .build();
        dataset.prepare();
        return dataset;
    }

    // Define the model name
    public static final String MODEL_NAME = "shoeclassifier";

    // Create and return the shoe classification model
    public static ai.djl.Model getModel(String name) {
        ai.djl.Model model = ai.djl.Model.newInstance(MODEL_NAME);
        Block resNet50 = ResNetV1.builder()
                .setImageShape(new Shape(3, 224, 224))
                .setNumLayers(50)
                .setOutSize(10)
                .build();
        model.setBlock(resNet50);
        return model;
    }

    // Setup the training configuration with the specified loss function
    public static TrainingConfig setupTrainingConfig(Loss loss) {
        return new DefaultTrainingConfig(loss)
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());
    }

    // Train the model using the provided training and validation datasets
    public static void train(RandomAccessDataset trainDataset, RandomAccessDataset validateDataset,
            TrainingConfig config, String name) throws IOException, TranslateException {
        try (ai.djl.Model model = getModel(name);
                Trainer trainer = model.newTrainer(config)) {
            trainer.setMetrics(new Metrics());
            Shape inputShape = new Shape(1, 3, 224, 224);
            trainer.initialize(inputShape);
            EasyTrain.fit(trainer, 1, trainDataset, validateDataset);
            TrainingResult result = trainer.getTrainingResult();
            model.setProperty("Epoch", String.valueOf(2));
            model.setProperty("Accuracy", String.format("%.5f", result.getValidateEvaluation("Accuracy")));
            model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
            model.save(Paths.get(Paths.get("src", "main", "model").toString()), MODEL_NAME);
        }
    }
}