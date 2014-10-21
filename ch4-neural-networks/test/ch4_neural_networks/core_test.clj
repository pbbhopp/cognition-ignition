(ns ch4-neural-networks.core-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.core :refer :all]))

(deftest make-neural-network-test
  (testing "making an initial state neural network"
    (let [nn [(make-layer 4 2) (make-layer 1 2)]]
      (is (= (count nn) 2)))))

(def nn (atom [[{:weights [0.014375367932126654 0.21631191689344542 0.7413991243079338]
                 :last-delta [0.0 0.0 0.0]
                 :deriv [0.0 0.0 0.0]}
                {:weights [-0.24575717337970066 -0.2671492428973788 0.15417318630866406]
                 :last-delta [0.0 0.0 0.0]
                 :deriv [0.0 0.0 0.0]}
                {:weights [-0.14575100638705296 -0.024118668071382188 -0.1322270911223906]
                 :last-delta [0.0 0.0 0.0]
                 :deriv [0.0 0.0 0.0]}
                {:weights [0.22431631387182704 -0.18362989197619173 -0.40702572197133036]
                 :last-delta [0.0 0.0 0.0]
                 :deriv [0.0 0.0 0.0]}]
               [{:weights [0.19869154577851822 0.06557770702187848 0.25770916351234513 -0.30487838476232276 -0.4954168557933254]
                 :last-delta [0.0 0.0 0.0 0.0 0.0]
                 :deriv [0.0 0.0 0.0 0.0 0.0]}]]))

(deftest forward-propagate-test
  (testing "forward propagate neural network"
    (let [_       (forward-propagate-net nn [0 0])
          outputs1 (map #(:output %) (first @nn))
          activs1  (map #(:activation %) (first @nn))
          outputs2 (map #(:output %) (second @nn))
          activs2  (map #(:activation %) (second @nn))]
      (is (= outputs1 [0.6299897352646393 0.15296315642714714 -0.13146182301041381 -0.38594433967392094]))
      (is (= activs1 [0.7413991243079338 0.15417318630866406 -0.1322270911223906 -0.40702572197133036]))
      (is (= outputs2 [-0.26959319163346745]))
      (is (= activs2 [-0.27642507796554894])))))

(deftest calculate-output-error
  (testing "find delta or diff between output and expected output"
    (let [_     (output-error nn 0)
          error (get-in @nn [1 0 :delta])]
      (is (= error 0.24999902664118395)))))

(deftest calculate-sum-of-errors-in-weights
  (testing "find sum of errors in weights"
    (let [sum-of-errors-in-weights (sum-errors 0 (second @nn))]
      (is (= sum-of-errors-in-weights 0.0496726930464618)))))


