package com.schmij03.sw10.playground.controller;

import ai.djl.Application;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.util.Utils;
import org.springframework.stereotype.Controller;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import com.schmij03.sw10.playground.Models;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Map;

@Controller
public class PredictionController {

    private static final String UPLOAD_DIR = "upload-dir"; // Directory to store uploaded images
    private static final String MODEL_NAME = "model/shoeclassifier-0002.params"; // Path to the trained model

    @PostMapping("/predict")
    public String handlePredictionRequest(@RequestParam("image") MultipartFile file, Map<String, Object> model) {
        if (file.isEmpty()) {
            model.put("error", "Please select an image file to upload.");
            return "index";
        }

        try {
            String fileName = StringUtils.cleanPath(file.getOriginalFilename());
            Path uploadPath = Path.of(UPLOAD_DIR).toAbsolutePath().normalize();
            Path filePath = uploadPath.resolve(fileName);

            // Save the uploaded image to the server
            Files.createDirectories(uploadPath);
            Files.copy(file.getInputStream(), filePath, StandardCopyOption.REPLACE_EXISTING);

            // Load the trained model
            Model trainedModel = ModelZoo.loadModel(getModelCriteria());

            // Load the image and perform prediction
            Image image = ImageFactory.getInstance().fromFile(filePath);
            List<Classifications.Classification> predictions = predict(image, trainedModel);

            // Get the top prediction
            List<Classifications.Classification> top3Predictions = predictions.subList(0,
                    Math.min(1, predictions.size()));

            // Pass the predictions to the view
            model.put("fileName", fileName);
            model.put("predictions", top3Predictions);
        } catch (IOException | ModelException | TranslateException e) {
            model.put("error", "Failed to process the uploaded image: " + e.getMessage());
        }

        return "index";
    }

    @GetMapping("/ping")
    public String ping() {
        return "app is up and running";
    }

    private List<Classifications.Classification> predict(Image image, Model model) throws TranslateException {
        try (Predictor<Image, Classifications> predictor = model.newPredictor(getTranslator())) {
            return predictor.predict(image).items();
        }
    }

    private Translator<Image, Classifications> getTranslator() {
        return ImageClassificationTranslator.builder()
                .addTransform(new Resize(Models.IMAGE_WIDTH, Models.IMAGE_HEIGHT))
                .addTransform(new ToTensor())
                .optApplySoftmax(true)
                .build();
    }

    private Criteria<Image, Classifications> getModelCriteria() {
        return Criteria.builder()
                .setTypes(Image.class, Classifications.class)
                .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                .build();
    }
}
