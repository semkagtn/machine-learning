package com.semkagtn.machinelearning.homework1;

import com.semkagtn.machinelearning.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.Loader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.InputStream;
import java.util.*;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * Created by semkagtn on 15.09.15.
 */
public class KnnClassifier extends Classifier {

    private static double calculateZ(Instance instance) {
        double z = 0.0;
        for (int i = 0; i < instance.numAttributes(); i++) {
            if (i == instance.classIndex()) {
                continue;
            }
            z += instance.value(i) * instance.value(i);
        }
        return Math.exp(z);
    }

    private static BiFunction<Instance, Instance, Double> metric = (x, y) -> {
        double result = 0.0;
        for (int i = 0; i < x.numAttributes(); i++) {
            if (i == x.classIndex()) {
                continue;
            }
            double xi = x.value(i);
            double yi = y.value(i);
            result += (xi - yi) * (xi - yi);
        }
        double z = calculateZ(x) - calculateZ(y);
        result += z * z;
        return Math.sqrt(result);
    };

    private int k;
    private Instances instances;

    public KnnClassifier(int k) {
        this.k = k;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        this.instances = new Instances(instances);
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.setMinimumNumberInstances(1);
        return result;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        List<Instance> neighbours = IntStream.range(0, instances.numInstances())
                .mapToObj(instances::instance)
                .sorted((x, y) -> Double.compare(metric.apply(x, instance), metric.apply(y, instance)))
                .limit(k)
                .collect(Collectors.toList());
        Map<Double, Integer> nearest = new HashMap<>();
        for (Instance neighbour : neighbours) {
            double clazz = neighbour.classValue();
            if (!nearest.containsKey(clazz)) {
                nearest.put(clazz, 0);
            }
            nearest.put(clazz, nearest.get(clazz) + 1);
        }
        return nearest.entrySet().stream()
                .reduce((x, y) -> x.getValue() >= y.getValue() ? x : y)
                .get().getKey();
    }


    private static final String FILE_NAME = "KNN/chips.txt";
    private static final int FOLDS = 10;
    private static final int K = 50;

    public static void main(String[] args) throws Exception {
        Instances instances = Utils.readCsvFile(FILE_NAME);

        NumericToNominal filter = new NumericToNominal();
        filter.setOptions(new String[]{"-R", String.valueOf(instances.classIndex() + 1)});
        filter.setInputFormat(instances);
        instances = Filter.useFilter(instances, filter);

        double[] errors = new double[K];
        errors[0] = Double.MAX_VALUE;
        for (int k = 1; k < K; k++) {
            Classifier classifier = new KnnClassifier(k);
            Evaluation evaluation = new Evaluation(instances);
            evaluation.crossValidateModel(classifier, instances, FOLDS, new Random());
            errors[k] = evaluation.errorRate();
            System.out.println(k + "," + evaluation.errorRate());
        }
        System.out.println("BEST: " + DoubleStream.of(errors).min().getAsDouble());
    }
}
