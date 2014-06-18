(ns ch4-neural-networks.core-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.core :refer :all]))

(def training-input
  [[[0 0] [0]]
   [[0 1] [1]]
   [[1 0] [1]]
   [[1 1] [0]]])

(deftest make-neural-network-test
  (testing "making an initial state neural network"
    (let [nn (make-neural-network 2 2 1)]
      (is (= (:hidden-activ nn) [1 1 1]))
      (is (= (:input-activ nn) [1 1 1]))
      (is (= (:output-activ nn) [1])))))

(deftest activate-input-test
  (testing "input activation for single input node"
    (is (= (activate-input [1 1 1] [0 0]) [0 0 1]))))

(deftest activate-inputs-test
  (testing "input activation for many input nodes"
    (let [nn1 (make-neural-network 2 2 1)
          nn2 (activate-inputs nn1 training-input)]
      (is (= (:input-activ nn2) [[0 0 1] [0 1 1] [1 0 1] [1 1 1]])))))

(deftest activate-hidden-node-test
  (testing "find acivation value for a single hidden node"
    (is (= (activate-hidden-node [5 7 9] [1 0 1]) 0.9999999999986171))))