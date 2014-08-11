(ns ch6-naive-bayes.core)

(defn increment-feature [data feature category]
  (let [result  ((keyword feature) @data {category 0})
        updated (merge-with + {(keyword category) 1} result)]
    (swap! data assoc (keyword feature) updated)))

(defn feature-probability [data feature category]
  (let [result ((keyword feature) @data)
        sum    (reduce + (vals result))
        count  (category result)]             
    (if (= count 0)
      0
      (/ count sum))))

(defn category-probability [data category]
  (let [sum-coll (map #(reduce + (vals (second %))) @data)
        cat-coll (map #(category (second %) 0) @data)]
    (/ (reduce + cat-coll) (reduce + sum-coll))))

(defprotocol Feature
  (get-words [str]))

(extend-type String
  Feature
  (get-words [str] (vec (clojure.string/split str #" "))))

(defn train [classifier terms category]
  (doseq [word terms]
    (increment-feature classifier word category)))