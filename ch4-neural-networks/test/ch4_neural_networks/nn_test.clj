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
 (testing "should backward propagate neural network with correct updated weights"
  (let [nn (backward-propagate nn [0 0] [0] sigmoid dsigmoid)]
   (is (= nn [; hidden layer with updated weights
               [[2.0 2.0 -2.994281613967575]  ; 'and' neuron
               [2.0 2.0 -1.0248591356248118]] ; 'or' neuron
              ; output layer with updated weights
              [[-6.001000347835497 5.994327253232941 -3.0210928712214065]]])))))


(deftest xor-test
 (testing "should correctly solve XOR problem after training"
  (let [xy [[[0 0] [0]] [[0 1] [1]] [[1 0] [1]] [[1 1] [0]]]
        nn (reduce
             (fn [nn _] (reduce (fn [nn [x y]]
                                   (backward-propagate nn x y sigmoid dsigmoid))
                         nn xy))
             nn (range 5000))]
   (is (= (first (last (forward-propagate nn [0 0] sigmoid))) 0))
   (is (= (first (last (forward-propagate nn [0 1] sigmoid))) 1))
   (is (= (first (last (forward-propagate nn [1 0] sigmoid))) 1))
   (is (= (first (last (forward-propagate nn [1 1] sigmoid))) 0)))))
