(ns ch7-decision-trees.core-test
  (:require [clojure.test :refer :all]
            [ch7-decision-trees.core :refer :all]))
;
; this example was taken from the book "Machine Learning in Action" from Manning publications
; chapter 3 pages 39-48
;

(deftest entropy-test
  (testing "calculating Shannon entropy"
    (let [data [[1 1 "yes"] [1 1 "yes"] [1 0 "no"] [0 1 "no"] [0 1 "no"]]]
      (is (= (shannon-entropy data) 0.9709505944546686)))))

(deftest split-test
  (testing "split data set (vector of vectors) on axis"
    (let [data [[1 1 "yes"] [1 1 "yes"] [1 0 "no"] [0 1 "no"] [0 1 "no"]]]
      (is (= (split-data-set data 0 1) [[1 "yes"] [1 "yes"] [0 "no"]]))
      (is (= (split-data-set data 0 0) [[1 "no"] [1 "no"]])))))

(deftest best-feature-split-test
  (testing "best feature split on axis that gives best info gain"
    (let [data [[1 1 "yes"] [1 1 "yes"] [1 0 "no"] [0 1 "no"] [0 1 "no"]]]
      (is (= (find-best-feature-to-split data) 0)))))

(deftest build-decision-tree-test
  (testing "building decision btree"
    (let [data-set [[1 1 "yes"] [1 1 "yes"] [1 0 "no"] [0 1 "no"] [0 1 "no"]]
          feature-labels ["no surfacing" "flippers"]
          tree (build-tree data-set feature-labels)]
      (is (= tree {"no surfacing" {0 "no" 1 {"flippers" {0 "no" 1 "yes"}}}})))))
