package com.semkagtn.machinelearning.homework1.apriori;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Created by semkagtn on 01.10.15.
 */
public class Products implements Iterable<Product> {

    private List<Product> products;

    public Products(List<Product> products) {
        this.products = products;
    }

    public Products(Product product) {
        this.products = new ArrayList<>();
        this.products.add(product);
    }

    public Products(Products products, Product product) {
        this.products = new ArrayList<>();
        for (Product p : products) {
            this.products.add(p);
        }
        this.products.add(product);
    }

    public boolean contains(Product product) {
        return this.products.contains(product);
    }

    public boolean containsAll(Products other) {
        return this.products.containsAll(other.products);
    }

    public int size() {
        return this.products.size();
    }

    public Set<Integer> ids() {
        return this.products.stream().map(Product::getId).collect(Collectors.toSet());
    }

    public void removeIf(Predicate<Product> predicate) {
        this.products.removeIf(predicate);
    }

    @Override
    public Iterator<Product> iterator() {
        return products.iterator();
    }
}
