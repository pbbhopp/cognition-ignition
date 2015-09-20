(ns ch4-neural-networks.nn-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.layer :refer :all]
            [ch4-neural-networks.nn :refer :all]))

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (* -1.0 x)))))

(defn dsigmoid [y] (* y (- 1.0 y)))

(def l1 (->Layer [[1 1] [1 1]] [0.0 0.0] [0.0 0.0] [0.0 0.0] nil))
(def l2 (->Layer [[1 1] [1 1]] [0.0 0.0] [0.0 0.0] [0.0 0.0] nil))
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

(def l3 (->Layer [[0.1 0.2] [0.3 0.4] [0.5 0.6]] [0.0 0.0] [0.0 0.0] [0.0 0.0] nil))
(def l4 (->Layer [[0.5 0.2 0.3]] [0.0 0.0] [0.0 0.0] [0.0 0.0] nil))
(def nnet (->NN [l3 l4] 0.85))

(deftest real-test
  (testing "should train neural network"
    (let [x [[1 1] [1 0] [0 1] [0 0]] 
          y [[0] [1] [1] [0]]
	  n (reduce #(train %1 (first %2) (last %2) sigmoid dsigmoid) nnet (map #(list %1 %2) x y))]
      (is (= n [])))))

;
; Test case values were taken from http://www.generation5.org/content/2001/xornet.asp.
; You can go there to cross-validate test results below.
;

(deftest forward-propagate-test
  (testing "should forward propagate neural network with correct outputs and activations"
    (let [l1 (->Layer [[0.341232 0.129952 -0.923123] [-0.115223 0.570345 -0.328932]] [0 0 0] [0 0 0] [0 0 0] [1 0 0])
          l2 (->Layer [[-0.993423 0.164732 0.752621]] [0 0 0] [0 0 0] [0 0 0] [1 0 0])
          nn (->NN [l1 l2] 0.2)
           x [0 0] 
          nn (forward-prop nn x sigmoid)]
      (is (= (:activations (first (:layers nn))) [0.5844897593895427 0.4712260773225621])))))
