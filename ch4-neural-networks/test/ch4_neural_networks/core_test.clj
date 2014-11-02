(ns ch4-neural-networks.core-test
  (:require [clojure.test :refer :all]
            [ch4-neural-networks.core :refer :all]))

(def weights [[ 0.014375367932126654  0.216311916893445420  0.74139912430793380] 
               [-0.245757173379700660 -0.267149242897378800  0.15417318630866406]
               [-0.145751006387052960 -0.024118668071382188 -0.13222709112239060]
               [ 0.224316313871827040 -0.183629891976191730 -0.40702572197133036]])

(deftest recursive-forward-propagate-test
  (testing "recursively forward propagate neural network"
    (let [results     (forward weights [0 0])
          activations (:activations results)
          outputs     (:outputs results)]
      (is (= activations [0.7413991243079338 0.15417318630866406 -0.1322270911223906 -0.40702572197133036]))
      (is (= outputs [0.6299897352646393 0.15296315642714714 -0.13146182301041381 -0.38594433967392094])))))

(defn make-net []
  [{:weights [[0.014375367932126654 0.21631191689344542 0.7413991243079338]
              [-0.24575717337970066 -0.2671492428973788 0.15417318630866406]
              [-0.14575100638705296 -0.024118668071382188 -0.1322270911223906]
              [0.22431631387182704 -0.18362989197619173 -0.40702572197133036]]
    :last-delta [[0.0 0.0 0.0]
                 [0.0 0.0 0.0]
                 [0.0 0.0 0.0]
                 [0.0 0.0 0.0]]
    :deriv [[0.0 0.0 0.0]
            [0.0 0.0 0.0]
            [0.0 0.0 0.0]
            [0.0 0.0 0.0]]}          
   {:weights [[0.19869154577851822 0.06557770702187848 0.25770916351234513 -0.30487838476232276 -0.4954168557933254]]
    :last-delta [[0.0 0.0 0.0 0.0 0.0]]
    :deriv [[0.0 0.0 0.0 0.0 0.0]]}])

(deftest forward-propagate-test
  (testing "should forward propagate neural network with coreect outputs and activations"
    (let [nn       (make-net)
          -nn      (forward-propagate nn [0 0])
          outputs1 (:outputs (first -nn))
          activs1  (:activations (first -nn))
          outputs2 (:outputs (second -nn))
          activs2  (:activations (second -nn))]
      (is (= outputs1 [0.6299897352646393 0.15296315642714714 -0.13146182301041381 -0.38594433967392094]))
      (is (= activs1 [0.7413991243079338 0.15417318630866406 -0.1322270911223906 -0.40702572197133036]))
      (is (= outputs2 [-0.26959319163346745]))
      (is (= activs2 [-0.27642507796554894])))))
   
(deftest backward-propagate-test
  (testing "should backward propagate neural network with correct deltas"
    (let [nn       (make-net)
          -nn      (forward-propagate nn [0 0])
          --nn     (backward-propagate -nn 0)
          deltas1 (:deltas (first --nn))
          deltas2 (:deltas (last --nn))]
      (is (= deltas1 [0.029958243616168813 0.01601077209310136 0.06331359834046521 -0.06486620358573723]))
      (is (= deltas2 [0.24999902664118395])))))
   