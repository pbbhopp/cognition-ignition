(ns ch4-neural-networks.core-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.core :refer :all]))

(deftest make-neural-network-test
  (testing "making an initial statw neural network"
    (let [nn (make-neural-network 2 2 1)]
      (is (= (:hidden-activ nn) [1 1]))
      (is (= (:input-activ nn) [1 1]))
      (is (= (:output-activ nn) [1])))))
