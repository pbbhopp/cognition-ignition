(ns ch4-neural-networks.nn-test
 (:require [clojure.test :refer :all]
           [ch4-neural-networks.nn :refer :all])
 (:use [clojure.pprint]))

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (* -1.0 x)))))

(defn dsigmoid [y] (* y (- 1.0 y)))

(def nn [; hidden layer
         [[2 2 -3] ; 'and' neuron
         [2 2 -1]] ; 'or' neuron
         ; output layer
         [[-6 6 -3]]])

(deftest forward-propagate-test
  (testing "should forward propagate neural network with correct output"
    (let [outputs (forward-propagate nn [0.0 0.0] sigmoid)]
      (is (= (first (last outputs)) 0.3019490083227375)))))

(deftest backward-propagate-test
 (testing "should backward propagate neural network with correct deltas"
  (let [nn (backward-propagate nn [0 0] [0] sigmoid dsigmoid)]
   (is (= nn [; hidden layer with updated weights
               [[2.0 2.0 -2.994281613967575]  ; 'and' neuron
               [2.0 2.0 -1.0248591356248118]] ; 'or' neuron
              ; output layer with updated weights
              [[-6.001000347835497 5.994327253232941 -3.0210928712214065]]])))))
