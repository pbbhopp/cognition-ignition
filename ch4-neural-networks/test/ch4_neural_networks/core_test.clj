(ns ch4-neural-networks.core-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.core :refer :all]))

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

