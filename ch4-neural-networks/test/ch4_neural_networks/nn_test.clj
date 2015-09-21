(ns ch4-neural-networks.nn-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.layer :refer :all]
            [ch4-neural-networks.nn :refer :all]))

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (* -1.0 x)))))

(defn dsigmoid [y] (* y (- 1.0 y)))

;
; Test case values were taken from http://www.generation5.org/content/2001/xornet.asp.
; You can go there to cross-validate test results below.
;

(deftest forward-propagate-test
  (testing "should forward propagate neural network with correct outputs and activations"
    (let [l1 (->Layer [[0.341232 0.129952 -0.923123] [-0.115223 0.570345 -0.328932]] [0 0 0] [0 0 0] [0 0 0] [1 0 0])
          l2 (->Layer [[-0.993423 0.164732 0.752621]] [0 0 0] [0 0 0] [0 0 0] [1 0 0])
          nn (->NN [l1 l2] 0.5)
           x [0 0] 
          nn (forward-prop nn x sigmoid)]
      (is (= (:activations (first (:layers nn))) [0.5844897593895427 0.4712260773225621]))
      (is (= (:activations (last (:layers nn))) [0.3676098854895219])))))

(deftest backward-propagate-test
  (testing "should backward propagate neural network with correct delta calculations"
    (let [l1 (->Layer [[0.341232 0.129952 -0.923123] [-0.115223 0.570345 -0.328932]] [0 0 0] [0 0 0] [0 0 0] [1 0 0])
          l2 (->Layer [[-0.993423 0.164732 0.752621]] [0 0 0] [0 0 0] [0 0 0] [1 0 0])
          nn (->NN [l1 l2] 0.5)
           x [0 0]
          nn (forward-prop nn x sigmoid)
          l2 (last (:layers nn))
          l2 (backprop l2 [0.3676098854895219] dsigmoid)]
      (is (= (:deltas l2) [-0.0848972546030838])))))
