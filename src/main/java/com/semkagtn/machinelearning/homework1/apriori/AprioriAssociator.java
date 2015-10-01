package com.semkagtn.machinelearning.homework1.apriori;

import com.semkagtn.machinelearning.Utils;
import weka.associations.AbstractAssociator;
import weka.associations.Associator;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by semkagtn on 29.09.15.
 */
public class AprioriAssociator extends AbstractAssociator {

    private double minSupport;
    private double minConfidence;
    private int numTransactions;
    private Products products;
    private List<Products> transactions;
    private List<Products> conjunctions;

    public AprioriAssociator(double minSupport, double minConfidence) {
        this.minSupport = minSupport;
        this.minConfidence = minConfidence;
    }

    private void buildProducts(Instances instances) {
        List<Product> productList = new ArrayList<>();
        Enumeration<?> productsEnumeration = instances.attribute(0).enumerateValues();
        while (productsEnumeration.hasMoreElements()) {
            String productName = (String) productsEnumeration.nextElement();
            Product product = Product.get(productName);
            double support = calculateSupport(new Products(product));
            if (support >= minSupport) {
                productList.add(product);
            }
        }
        products = new Products(productList);
    }

    private void buildTransactions(Instances instances) {
        Map<Long, Set<Product>> transactionProducts = new HashMap<>();
        for (int i = 0; i < instances.numInstances(); i++) {
            Instance instance = instances.instance(i);
            String stringTransaction = instance.stringValue(instance.classIndex());
            long transactionId = Integer.valueOf(stringTransaction.substring(1, stringTransaction.length() - 1));
            String productName = instance.stringValue(0);
            Product product = Product.get(productName);
            if (!transactionProducts.containsKey(transactionId)) {
                transactionProducts.put(transactionId, new HashSet<>());
            }
            transactionProducts.get(transactionId).add(product);
        }
        numTransactions = transactionProducts.size();
        transactions = transactionProducts.values().stream()
                .map(set -> new Products(new ArrayList<>(set)))
                .collect(Collectors.toList());
    }

    private void optimizeTransactions() {
        for (Products transction : transactions) {
            transction.removeIf(product -> !products.contains(product));
        }
        transactions = transactions.stream()
                .filter(products -> products.size() >= 2)
                .collect(Collectors.toList());
    }

    private double calculateSupport(Products conjunction) {
        int support = 0;
        for (Products transaction : transactions) {
            if (transaction.containsAll(conjunction)) {
                support++;
            }
        }
        return (double) support / numTransactions;
    }

    private void buildConjunctions() {
        conjunctions = new ArrayList<>();
        List<Products> previousConjunctionsSubset = new ArrayList<>();
        for (Product product : products) {
            Products conjunction = new Products(product);
            previousConjunctionsSubset.add(conjunction);
        }
        for (int i = 0; i < products.size() - 1; i++) {
            List<Products> conjunctionsSubsetResult = new ArrayList<>();
            for (Products conjunction : previousConjunctionsSubset) {
                Set<Integer> ids = conjunction.ids();
                for (Product product : products) {
                    if (ids.contains(product.getId())) {
                        continue;
                    }
                    Products newConjunction = new Products(conjunction, product);
                    double support = calculateSupport(newConjunction);
                    if (support >= minSupport) {
                        conjunctionsSubsetResult.add(newConjunction);
                    }
                }
            }
            if (conjunctionsSubsetResult.isEmpty()) {
                return;
            }
            previousConjunctionsSubset = conjunctionsSubsetResult;
            conjunctions.addAll(conjunctionsSubsetResult);
            System.out.println();
        }
    }

    @Override
    public void buildAssociations(Instances instances) throws Exception {
        buildTransactions(instances);
        buildProducts(instances);
        optimizeTransactions();
        buildConjunctions();
    }

    @Override
    public String toString() {
        return super.toString();
    }


    private static final String FILE_NAME = "Association_rules/supermarket.arff";
    private static final double MIN_SUPPORT = 0.003;
    private static final double MIN_CONFIDENCE = 0.1;

    public static void main(String[] args) throws Exception {
        Instances instances = Utils.readArffFile(FILE_NAME);

        NumericToNominal numericToNominal = new NumericToNominal();
        numericToNominal.setOptions(new String[]{"-R", String.valueOf(instances.classIndex() + 1)});
        numericToNominal.setInputFormat(instances);
        instances = Filter.useFilter(instances, numericToNominal);

        Associator associator = new AprioriAssociator(MIN_SUPPORT, MIN_CONFIDENCE);
        associator.buildAssociations(instances);
        System.out.println();

//        AssociatorEvaluation evaluation = new AssociatorEvaluation();
//        evaluation.evaluate(associator, instances);
//        System.out.println(evaluation.toSummaryString());
    }
}
