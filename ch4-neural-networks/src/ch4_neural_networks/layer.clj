(ns ch4-neural-networks.layer)
  
(defn feed [w x f]
  (let [coll (map #(reduce + (map (fn [a b] (* a b)) %1 %2)) w x)]
    (map f coll)))

(defn backprop [w v s df]
  (let [err (map #(* %1 %2) s (map df v))]
    (map #(reduce + (map * err %)) w)))
