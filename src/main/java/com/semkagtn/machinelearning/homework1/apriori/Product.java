package com.semkagtn.machinelearning.homework1.apriori;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by semkagtn on 01.10.15.
 */
public class Product implements Comparable<Product> {

    private static int lastId = 0;
    private static Map<String, Product> products = new HashMap<>();

    public static Product get(String name) {
        if (!products.containsKey(name)) {
            products.put(name, new Product(++lastId, name));
        }
        return products.get(name);
    }

    private int id;
    private String name;

    private Product(int id, String name) {
        this.id = id;
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public int getId() {
        return id;
    }

    @Override
    public int compareTo(Product other) {
        return Integer.compare(this.getId(), other.getId());
    }
}
