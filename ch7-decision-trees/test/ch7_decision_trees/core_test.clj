(ns ch7-decision-trees.core-test
  (:require [clojure.test :refer :all]
            [ch7-decision-trees.core :refer :all]))

(deftest a-test
  (testing "Shannon entropy"
    (let [data [[1 1 "yes"] [1 1 "yes"] [1 0 "no"] [0 1 "no"] [0 1 "no"]]]
      (is (= (shannon-entropy data) 0.9709505944546686)))))
