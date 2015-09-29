package com.semkagtn.machinelearning.homework1;

import com.semkagtn.machinelearning.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.util.Random;

/**
 * Created by semkagtn on 29.09.15.
 */
public class LogisticRegressionClassifier extends Classifier {

    private double[] coefs;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        this.coefs = new double[data.numAttributes()];
        gradientAscent(data);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (logisticFunction(linearFunction(instance)) < 0.5) {
            return 0.0;
        }
        return 1.0;
    }

    private static final double STEP = 0.001;
    private static final double PRECISION = 0.01;
    private static final double REGULARIZATION_COEF = 0.1;

    private void gradientAscent(Instances instances) throws Exception {
        double[] newCoefs = new double[coefs.length];
        do {
            System.arraycopy(newCoefs, 0, coefs, 0, coefs.length);
            for (int i = 0; i < coefs.length; i++) {
                newCoefs[i] = coefs[i] + STEP * derivative(i, instances);
            }
        } while (norm(newCoefs) > PRECISION);
        coefs = newCoefs;
    }

    private double norm(double[] oldCoefs) {
        double result = 0.0;
        for (int i = 0; i < coefs.length; i++) {
            result += (oldCoefs[i] - coefs[i]) * (oldCoefs[i] - coefs[i]);
        }
        return Math.sqrt(result);
    }

    private double derivative(int k, Instances instances) throws Exception {
        double result = 0.0;
        for (int i = 0; i < instances.numInstances(); i++) {
            Instance instance = instances.instance(i);
            double kValue = k != instance.classIndex() ? instance.value(k) : 1;
            double y = instance.classValue();
            double z = linearFunction(instance);
            result += (y - logisticFunction(z)) * kValue + REGULARIZATION_COEF * kValue;
        }
        return result;
    }

    private double linearFunction(Instance instance) {
        double result = 0.0;
        for (int i = 0; i < coefs.length; i++) {
            double iValue = i != instance.classIndex() ? instance.value(i) : 1;
            result += coefs[i] * iValue;
        }
        return result;
    }

    private static double logisticFunction(double z) {
        return 1.0 / (1 + Math.exp(-z));
    }


    private static final String FILE_NAME = "Logistic_regression/chips-new.txt";
    private static final int FOLDS = 5;

    public static void main(String[] args) throws Exception {
        Instances instances = Utils.readCsvFile(FILE_NAME);

        NumericToNominal filter = new NumericToNominal();
        filter.setOptions(new String[]{"-R", String.valueOf(instances.classIndex() + 1)});
        filter.setInputFormat(instances);
        instances = Filter.useFilter(instances, filter);

        Filter normalize = new Normalize();
        normalize.setInputFormat(instances);
        instances = Filter.useFilter(instances, normalize);

        Classifier classifier = new LogisticRegressionClassifier();
        classifier.buildClassifier(instances);

        Evaluation evaluation = new Evaluation(instances);
        evaluation.crossValidateModel(classifier, instances, FOLDS, new Random());
        System.out.println(evaluation.toSummaryString());
    }
}
