(ns ch4-neural-networks.nn-test
 (:require [clojure.test :refer :all]
           [ch4-neural-networks.nn :refer :all])
 (:use [clojure.pprint]))

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (* -1.0 x)))))

(defn dsigmoid [y] (* y (- 1.0 y)))

(def nn [
          [[0.02062429003326055 0.3435545438096753  0.5704151456736803]
           [0.10871857288071629 0.4449524922353063 -0.1832750222077525]]

          [[-0.7977182602187687 0.1718961033660047 -0.06964779721310455]]])

(deftest forward-propagate-test
  (testing "should forward propagate neural network with correct output"
    (let [outputs (forward-propagate nn [0.0 0.0] sigmoid)]
      (is (= (first (last outputs)) 0.3019490083227375)))))

(deftest backward-propagate-test
 (testing "should backward propagate neural network with correct deltas"
  (let [nn (backward-propagate nn [0 0] [0] sigmoid dsigmoid)]
   (is (= nn [])))))
