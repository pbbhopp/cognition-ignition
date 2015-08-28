
(ns ch4-neural-networks.nn-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.nn :refer :all]))

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (* -1.0 x)))))

(defn dsigmoid [y] (* y (- 1.0 y)))

(def nn (make-nn [[2 2] [2 3]] 0.1))

(deftest forward-prop-test
  (testing "should forward propogate neural network"
    (let [x [[1 1] [1 1]]]
      (is (= (forward-prop nn x sigmoid) '(-4.0 -8.0))))))

