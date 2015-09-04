(ns ch4-neural-networks.nn-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.layer :refer :all]
            [ch4-neural-networks.nn :refer :all]))

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (* -1.0 x)))))

(defn dsigmoid [y] (* y (- 1.0 y)))

(def l1 (->Layer [[1 1] [1 1]] [0.0 0.0] [0.0 0.0] [0.0 0.0]))
(def l2 (->Layer [[1 1] [1 1]] [0.0 0.0] [0.0 0.0] [0.0 0.0]))
(def nn (->NN [l1 l2] 0.1))

(deftest forward-prop-test
  (testing "should forward propogate neural network"
    (let [x [1 1] 
          v 0.8534092045709026
          n (forward-prop nn x sigmoid)]
	  (is (= (:activations (last (:layers n))) (list v v))))))

(deftest train-test
  (testing "should train neural network"
    (let [x [1 1] 
          y [1]
          v 0.8534092045709026
	  n (train nn x y sigmoid dsigmoid)]
      (is (= n [])))))

(def l3 (->Layer [[0.1 0.2] [0.3 0.4] [0.5 0.6]] [0.0 0.0 0.0] [0.0 0.0 0.0] [0.0 0.0 0.0]))
(def l4 (->Layer [[0.5 0.2 0.3]] [0.0] [0.0] [0.0]))
(def nnet (->NN [l3 l4] 0.25))

(deftest real-test
  (testing "should train neural network"
    (let [x [[1 1] [1 0] [0 1] [0 0]] 
          y [[0] [1] [1] [0]]
	  n (reduce #(train %1 (first %2) (last %2) sigmoid dsigmoid) nnet (map #(list %1 %2) x y))]
      (is (= n [])))))
