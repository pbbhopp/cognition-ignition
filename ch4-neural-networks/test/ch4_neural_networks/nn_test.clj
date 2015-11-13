(ns ch4-neural-networks.nn-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.nn :refer :all])
  (:use [clojure.pprint]))

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (* -1.0 x)))))

(defn dsigmoid [y] (* y (- 1.0 y)))

(def nn [[{:weights [0.02062429003326055 0.3435545438096753 0.5704151456736803]
           :last_delta [0.0 0.0 0.0] :deriv [0.0 0.0  0.0] :activation 0.5704151456736803 :output 0.6388589623271096}
          {:weights [0.10871857288071629  0.4449524922353063  -0.1832750222077525]
           :last_delta [0.0 0.0 0.0] :deriv [0.0 0.0 0.0] :activation -0.1832750222077525 :output 0.45430906842459423}
          {:weights [0.11848174842917547 0.01402578209170834 0.4465007733452825]
           :last_delta [0.0 0.0 0.0] :deriv [0.0 0.0 0.0] :activation 0.4465007733452825 :output 0.6098069400845292}
          {:weights [-0.42056743700655  0.14937038322015017  -0.098205682489091]
           :last_delta [0.0 0.0 0.0] :deriv [0.0 0.0 0.0] :activation -0.098205682489091 :output 0.47546829225302856}]
         [{:weights [-0.7977182602187687 0.1718961033660047 -0.06964779721310455 -0.1943098215433533 -0.23595593944507165]
           :last_delta [0.0 0.0 0.0 0.0 0.0] :deriv [0.0 0.0 0.0 0.0 0.0] :activation -0.8023513099311541 :output 0.3095227756800094}]])

(deftest forward-propagate-test
  (testing "should forward propagate neural network with correct output"
    (let [output (forward-propagate nn [0.0 0.0] sigmoid)]
      (is (= output 0.3095227756800094)))))

(deftest backward-propagate-test
  (testing "should backward propagate neural network with correct deltas"
    (let [nn (backward-propagate nn 0.0 dsigmoid)]
      (is (= (:delta (first (last nn))) -0.06615072074375726))
      (is (= (mapv :delta (first nn)) [0.012174915260063871 -0.002819023880106377 0.0010962607598879325 0.003205698247883244])))))

(deftest calc-err-derivatives-for-weights-test
  (testing "should backward propagate neural network with correct deltas"
    (let [nn (calc-err-derivative nn [0.0 0.0])]
      (is (= (:deriv (first (last nn))) [-0.042260980811547166 -0.030052872316711842 -0.04033916860113681 -0.031452570223341254 -0.06615072074375726]))
      (is (= (mapv :delta (first nn)) [[0.0 0.0 0.012174915260063871] [0.0 0.0 -0.002819023880106377] [0.0 0.0 0.0010962607598879325] [0.0 0.0 0.003205698247883244]])))))
