package com.semkagtn.machinelearning;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.Loader;

import java.io.IOException;
import java.io.InputStream;

/**
 * Created by semkagtn on 15.09.15.
 */
public class Utils {

    public static Instances readCsvFile(String fileName) {
        Instances instances;
        try {
            InputStream inputStream = Thread.currentThread().getContextClassLoader().getResourceAsStream(fileName);
            Loader loader = new CSVLoader();
            loader.setSource(inputStream);

            instances = loader.getDataSet();
            instances.setClassIndex(instances.numAttributes() - 1);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return instances;
    }

    private Utils() {

    }
}
