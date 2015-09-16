package com.semkagtn.machinelearning.homework1;

import com.semkagtn.machinelearning.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Standardize;

import java.util.Random;

/**
 * Created by semkagtn on 15.09.15.
 */
public class LinearRegressionClassifier extends Classifier {

    private double[] coefs;

    public LinearRegressionClassifier(int dimension) {
        this.coefs = new double[dimension + 1];
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        gradientDescent(data);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double result = 0.0;
        for (int i = 0; i < coefs.length; i++) {
            double iValue = i != instance.classIndex() ? instance.value(i) : 1;
            result += coefs[i] * iValue;
        }
        return result;
    }

    private static final double STEP = 0.0001;
    private static final double PRECISION = 5;
    private static final double REGULARIZATION_COEF = 5;

    private void gradientDescent(Instances instances) throws Exception {
        double[] newCoefs = new double[coefs.length];
        do {
            System.arraycopy(newCoefs, 0, coefs, 0, coefs.length);
            for (int i = 0; i < coefs.length; i++) {
                newCoefs[i] = coefs[i] - STEP * derivative(i, instances);
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
            result += (classifyInstance(instance) - instance.classValue()) * kValue
                    + REGULARIZATION_COEF * kValue;
        }
        return 2 * result;
    }


    private static final String FILE_NAME = "Linear_regression/prices.txt";
    private static final int FOLDS = 5;

    public static void main(String[] args) throws Exception {
        Instances instances = Utils.readCsvFile(FILE_NAME);

        Filter filter = new Normalize();
        filter.setInputFormat(instances);
        instances = Filter.useFilter(instances, filter);

        Classifier classifier = new LinearRegressionClassifier(instances.numAttributes() - 1);
        classifier.buildClassifier(instances);

        Evaluation evaluation = new Evaluation(instances);
        evaluation.crossValidateModel(classifier, instances, FOLDS, new Random());
        System.out.println(evaluation.toSummaryString());
    }
}
