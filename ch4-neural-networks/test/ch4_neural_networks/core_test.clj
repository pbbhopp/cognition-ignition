(ns ch4-neural-networks.core-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.core :refer :all]))
;
; Test case values were taken from http://www.generation5.org/content/2001/xornet.asp.
; You can go there to cross-validate test results below.
;

(deftest forward-propagate-test
  (testing "should forward propagate neural network with correct outputs and activations"
    (let [nn [[{:weights [0.129952 -0.923123  0.341232] :last-delta [0 0] :deriv [0 0]}
               {:weights [0.570345 -0.328932 -0.115223] :last-delta [0 0] :deriv [0 0]}]
              [{:weights [0.164732  0.752621 -0.993423] :last-delta [0 0] :deriv [0 0]}]]
          nn (forward-propagate nn [0 0])
          n1 (ffirst nn)
          n2 (second (first nn))
          n3 (first (second nn))]
      (is (=  0.3412320000000000 (:activation n1)))
      (is (=  0.5844897593895427 (:output n1)))
      (is (= -0.1152230000000000 (:activation n2)))
      (is (=  0.4712260773225621 (:output n2)))
      (is (= -0.5424841914156577 (:activation n3)))
      (is (=  0.3676098854895219 (:output n3))))))

(deftest backward-propagate-test
  (testing "should backward propagate neural network with correct delta calculations"
    (let [nn [[{:weights [0.129952 -0.923123  0.341232] :output 0.584490}
               {:weights [0.570345 -0.328932 -0.115223] :output 0.471226}]
              [{:weights [0.164732  0.752621 -0.993423] :output 0.367610}]]
          expected 0.0    
          nn (backward-propagate nn expected)
          d1 (:delta (ffirst nn))
          d2 (:delta (second (first nn)))
          d3 (:delta (first (second nn)))]
      (is (= d1 -0.0034189768826391386))
      (is (= d2 -0.016026374866587624))
      (is (= d3 -0.085459358320919)))))

