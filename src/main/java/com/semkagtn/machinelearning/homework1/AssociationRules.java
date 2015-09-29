package com.semkagtn.machinelearning.homework1;

import com.semkagtn.machinelearning.Utils;
import weka.associations.Apriori;
import weka.associations.Associator;
import weka.associations.AssociatorEvaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 * Created by semkagtn on 29.09.15.
 */
public class AssociationRules {

    private static final String FILE_NAME = "Association_rules/supermarket.arff";

    public static void main(String[] args) throws Exception {
        // Weka implementation

        Instances instances = Utils.readArffFile(FILE_NAME);

        NumericToNominal numericToNominal = new NumericToNominal();
        numericToNominal.setOptions(new String[]{"-R", String.valueOf(instances.classIndex() + 1)});
        numericToNominal.setInputFormat(instances);
        instances = Filter.useFilter(instances, numericToNominal);

        Associator associator = new Apriori();
        associator.buildAssociations(instances);

        AssociatorEvaluation evaluation = new AssociatorEvaluation();
        evaluation.evaluate(associator, instances);
        System.out.println(evaluation.toSummaryString());
    }
}
