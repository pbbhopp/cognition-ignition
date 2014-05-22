(ns ch3-discovering-groups.core-test
  (:require [clojure.test :refer :all]
            [ch3-discovering-groups.core :refer :all])
  (:use     [clojure.string :as str]
            [clojure.data.csv :only (read-csv)]
            [clojure.java.io :as io]))

(defn make-blog-data []
  {:words [] :blogs [] :word-counts []})

(defn strings->nums [coll]
  (map read-string coll))

(defn append-data [blog-data line]
  (let [blog-name  (first line)
        counts     (strings->nums (rest line))]
    (merge-with concat blog-data {:blogs [blog-name] :word-counts counts})))

(defn make-data [file]
  (let [raw   (with-open [in-file (clojure.java.io/reader file)]
                (doall (clojure.data.csv/read-csv in-file)))
        head  (first raw)
        tally (rest raw)
        data  (reduce
                append-data
                (make-blog-data)
                tally)]
    (assoc data :words head)))

(def data (make-data "test/ch3_discovering_groups/data.csv"))

(def v1
  [0.0 1.0 0.0 0.0 3.0 3.0 0.0 0.0 3.0 0.0 6.0 0.0 1.0 0.0 4.0 3.0 0.0 0.0 0.0 0.0 0.0 4.0 0.0])
(def v2 
  [0.0 2.0 1.0 0.0 6.0 2.0 1.0 0.0 4.0 5.0 25.0 0.0 0.0 0.0 6.0 12.0 4.0 2.0 1.0 4.0 0.0 3.0 0.0])

(deftest pearson-test
  (testing "Pearson score"
    (is (= (pearson v1 v2) 0.25004925261947253))))

(deftest kmeans-test
  (testing "kmeans clustering"
    (is (= (take 5 (kmeans (:word-counts data))) [0.7408868896992121 0.31305486463589927 3.992773202381228 20.445880868103867 15.259185101093632]))))