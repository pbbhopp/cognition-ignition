(ns ch4-neural-networks.nn-test
 (:require [clojure.test :refer :all]
           [ch4-neural-networks.nn :refer :all])
 (:use [clojure.pprint]))

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (* -1.0 x)))))

(defn dsigmoid [y] (* y (- 1.0 y)))

(def nn [[[0.02062429003326055 0.3435545438096753  0.5704151456736803]
          [0.10871857288071629 0.4449524922353063 -0.1832750222077525]
          [0.11848174842917547 0.01402578209170834 0.4465007733452825]]
         [[-0.7977182602187687 0.1718961033660047 -0.06964779721310455 -0.1943098215433533 -0.23595593944507165]]])

(deftest forward-propagate-test
  (testing "should forward propagate neural network with correct output"
    (let [outputs (forward-propagate nn [0.0 0.0] sigmoid)]
      (is (= (first (last outputs)) 0.3019490083227375)))))

(deftest backward-propagate-test
 (testing "should backward propagate neural network with correct deltas"
  (let [nn (backward-propagate nn 0.0 dsigmoid)]
   (is (= (:delta (first (last nn))) -0.06615072074375726))
   (is (= (mapv :delta (first nn)) [0.012174915260063871 -0.002819023880106377 0.0010962607598879325 0.003205698247883244])))))

(def n2 [[[0.02062429003326055 0.3435545438096753 0.5704151456736803]
          [0.10871857288071629 0.4449524922353063 -0.1832750222077525]
          [0.11848174842917547 0.01402578209170834 0.4465007733452825]
          [-0.42056743700655 0.14937038322015017 -0.098205682489091]]
         [[-0.7977182602187687 0.1718961033660047 -0.06964779721310455 -0.1943098215433533 -0.23595593944507165]]])


(deftest calc-err-derivatives-for-weights-test
 (testing "should correctly calculate error derivatives for weights"
  (let [nn (calc-err-derivatives n2 [0.0 0.0])]
   (is (= (:deriv (first (last nn))) [-0.042260980811547166 -0.030052872316711842 -0.04033916860113681 -0.031452570223341254 -0.06615072074375726]))
   (is (= (mapv :deriv (first nn)) [[0.0 0.0 0.012174915260063871] [0.0 0.0 -0.002819023880106377] [0.0 0.0 0.0010962607598879325] [0.0 0.0 0.003205698247883244]])))))

(def n3 [[[0.02062429003326055 0.3435545438096753 0.5704151456736803]
          [0.10871857288071629 0.4449524922353063 -0.1832750222077525]
          [0.11848174842917547 0.01402578209170834 0.4465007733452825]
          [-0.42056743700655 0.14937038322015017 -0.098205682489091]]
         [[-0.7977182602187687 0.1718961033660047 -0.06964779721310455 -0.1943098215433533 -0.23595593944507165]]])

(deftest update-weights-test
 (testing "should correctly update weights"
  (let [nn (update-weights n3 0.3 0.8)]
   (is (= (mapv :weights (first nn)) [[ 0.02062429003326055 0.3435545438096753   0.5740676202516996]
                                      [ 0.10871857288071629 0.4449524922353063  -0.1841207293717844]
                                      [ 0.11848174842917547 0.01402578209170834  0.4468296515732489]
                                      [-0.42056743700655    0.14937038322015017 -0.09724397301472604]]))
   (is (= (:weights (first (last nn))) [-0.8103965544622328 0.16288024167099113 -0.08174954779344559 -0.20374559261035569 -0.25580115566819883])))))

(deftest full-train-and-predict-test
  (testing "full training and prediction"
    (let [xor [[0 0 0] [0 1 1] [1 0 1] [1 1 0]]
          xy  (partition 2000 4 (take (* 4 2000) (cycle xor)))
          nn  (reduce #(train-network %1 %2 0.3 0.8 sigmoid dsigmoid) nn xy)]
    (is (= 0 (forward-propagate nn [0 0] sigmoid)))
    (is (= 1 (forward-propagate nn [0 1] sigmoid)))
    (is (= 1 (forward-propagate nn [1 0] sigmoid)))
    (is (= 0 (forward-propagate nn [1 1] sigmoid))))))