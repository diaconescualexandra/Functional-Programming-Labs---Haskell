-- lab 2
module Main where
import Data.Binary.Get (label)
import GHC.Lexeme (okSymChar)
import Control.Applicative (Applicative(liftA2))
import Distribution.Simple.Setup (trueArg)
import Distribution.Compat.Exception (tryIO)
import Control.Arrow (ArrowChoice(left))
import Text.XHtml (base, abbr, name, background)

-- poly :: Double -> Double -> Double -> Double -> Double
-- poly a b c x = a * x^2 + b * x + c

-- eeny :: Integer -> String
-- eeny n = if even n
--     then "eeny"
--     else "meeny"

-- cu if 
-- fizzbuzz :: Integer -> String
-- fizzbuzz n = if n `mod` 3 == 0 && n `mod` 5 == 0
--     then "FizzBuzz"
--     else if n `mod` 3 == 0
--         then "Fizz"
--     else if n `mod` 5 == 0
--         then "Buzz"
--     else
--         " "

-- cu garzi / conditii
-- fizzbuzz :: Integer -> String
-- fizzbuzz n
--     | n `mod` 3 == 0 && n `mod` 5 == 0 = "fizzbuzz"
--     | n `mod` 3 == 0 = "fizz"
--     | n `mod` 5 == 0 = "buzz"
--     | otherwise = ""


--recursivitate
-- tribonacciCazuri :: Integer -> Integer
-- tribonacciCazuri n
--     | n == 1 = 1
--     | n == 2 = 1
--     | n == 3 = 2
--     | n > 3 = tribonacciCazuri (n - 1) + tribonacciCazuri (n - 2) + tribonacciCazuri (n - 3)
--     | otherwise = error "Input invalid"

-- -- ecuational
-- tribonacciEcuational :: Integer -> Integer
-- tribonacciEcuational 1 = 1
-- tribonacciEcuational 2 = 1
-- tribonacciEcuational 3 = 2
-- tribonacciEcuational n =
--         tribonacciEcuational ( n-1 ) + tribonacciEcuational (n - 2) + tribonacciEcuational (n - 3)

-- CERINTA
-- B(n,k) = B(n-1,k) + B(n-1,k-1)
-- B(n,0) = 1
-- B(0,k) = 0

-- REZOLVARE
-- binomial :: Integer -> Integer -> Integer
-- binomial n k 
--     | k == 0 = 1
--     | n == 0 = 0
--     | otherwise = binomial( n-1) k + binomial (n-1) (k-1)

-- verifL - verifică dacă lungimea unei liste date ca parametru este pară.
-- verifL :: [Int] -> Bool
-- verifL l = even ( length l)

-- takefinal - pentru o listă l dată ca parametru s, i un număr n, întoarce o listă care cont, ine ultimele n
-- elemente ale listei l. Dacă lista are mai put, in de n elemente, întoarce lista nemodificată.
-- takefinal :: [Int] -> Int -> [Int]
-- takefinal l n = if length l >=n
--     then drop (length l - n) l
--     else
--         l
-- remove :: [Int] -> Int -> [Int]
-- remove l n = take(n-1) l ++ drop n l

-- Dată fiind o listă de numere întregi, să se scrie o funct, ie semiPareRec care elimină numerele impare
-- s, i le injumătăt,es, te pe cele pare. 

-- semiPareRec :: [Int] -> [Int]
-- semiPareRec [] = []
-- semiPareRec (h:t)
--     |even h = h `div` 2 : t'
--     |otherwise =t'
--     where t' = semiPareRec t

-- myreplicate :: Int -> Int -> [Int]
-- myreplicate n v 
--     | n<=0 = []
--     | otherwise =v : myreplicate (n-1) v

-- sumImp :: [Int] -> Int
-- sumImp [] =0
-- sumImp (h:t)
--     | odd h = h + ok
--     | otherwise = ok
--     where
--         ok = sumImp t

-- totalLen :: [String] -> Int
-- totalLen [] = 0
-- totalLen (h:t)
--     |head h == 'A' = length h + ok
--     |otherwise = ok
--     where
--         ok = totalLen t


-- lab 3
-- 1
-- palindrom :: String -> Bool
-- -- palindrom   = False
-- palindrom s = s == reverse s
    
-- vocale :: String -> Int
-- vocale s = length [ c | c <- s , c `elem` "aeiouAEIOU"]

-- nrVocale :: [String] -> Int
-- nrVocale [] =0
-- nrVocale (h:t)
--     |palindrom(h) == True = vocale h + nrvocale 
--     | otherwise = nrvocale
--     where nrvocale = nrVocale t

-- 2
-- f :: Int -> [Int] -> [Int]
-- f n [] = []
-- f n (h:t) 
--     | even h = h : n : f n t
--     | otherwise = h : f n t

--3
-- divizori :: Int -> [Int]
-- divizori n = [ x| x <- [1..n], n `mod` x == 0    ]

-- --4 
-- listadiv :: [Int] -> [[Int]]
-- listadiv l = [divizori c | c<-l]

-- --5 b)
-- inInterval :: Int -> Int -> [Int] -> [Int]
-- inInterval n m l = [ x | x<-l, x>=n && x<=m]

-- -- 5 a)
-- inIntervalRec :: Int -> Int -> [Int] -> [Int]
-- inIntervalRec _ _ [] = []
-- inIntervalRec n m (h : t)
--     | h >= n && h <= m = h:inIntervalRec n m t
--     | otherwise = inIntervalRec n m t


-- --  6 a)
-- pozitiveRec :: [Int] -> Int
-- pozitiveRec [] = 0
-- pozitiveRec (h:t)
--     |h > 0 = ok +1
--     | otherwise = ok
--     where ok = pozitiveRec t

-- pozitiveComp :: [Int] -> Int
-- pozitiveComp l =length [x | x<-l, x>0]

-- lab 4
-- 2
-- factori :: Int -> [Int]
-- factori n = [ x | x<- [1..n], n `mod` x == 0]

-- prim :: Int -> Bool
-- prim n = length (factori n) == 2  

-- numerePrime :: Int -> [Int]
-- numerePrime n = [x | x <- [2..n], prim x ]

--zip 
-- 5

-- myzip3 :: [Int] -> [Int] -> [Int] ->[(Int,Int,Int)]
-- myzip3 l b c = [(x,y,z) | (x,(y,z)) <-zip l (zip b c)]


-- 6
-- firstEl:: [(a,b)] -> [a]
-- firstEl pairs = map (\(x,_)->x) pairs

--7
-- sumList :: [[Int]] -> [Int]
-- sumList l = map (sum) l

--8
-- prel2 :: [Int] -> [Int]
-- prel2 l = ( map(\x -> x*2 ) (filter odd l) ++ map (\x -> x `div` 2) ( filter even l))

-- --9
-- f :: Char -> [String] ->[String]
-- f a s= filter (\str -> a `elem` str ) s

-- --10
-- li :: [Int] -> [Int]
-- li l = map (\x -> x^2) (filter  odd l)

-- --11
-- zl :: [Int] -> [Int]
-- zl l = map (\(index, value) -> value*value) (filter ( \(index, value) -> odd index) ( zip[0..] l))

-- --12
-- numaiVocale :: [String] -> [String]
-- numaiVocale l = map(filter (\chr -> chr `elem` "aeiouAEIOU") )l


-- lab 5

-- 1
-- sumaPatratelorImpare :: [Int] -> Int
-- sumaPatratelorImpare l = foldr (+) 0 ( map (\x -> x^2) (filter odd l))

-- -- 2
-- verifElem :: [Bool] -> Bool
-- verifElem l = foldr (&&) True l

-- -- 3
-- allVerifies :: (Int -> Bool) -> [Int] -> Bool
-- allVerifies prop l = foldr (&&) True ( map prop l)

-- 4
-- anyVerifies :: (Int -> Bool) -> [Int] -> Bool
-- anyVerifies prop l = foldr (||) False (map prop (filter prop l))

-- 5
-- mapFoldr :: (a -> b) -> [a] -> [b]
-- mapFoldr prop l = foldr (\x acc -> prop x : acc) [] l

-- filterFoldr :: (a -> Bool) -> [a] -> [a]
-- filterFoldr p xs = foldr (\x acc -> if p x then x : acc else acc) [] xs

-- 6
-- listToInt :: [Integer] -> Integer
-- listToInt = foldl ( \acc x -> acc *10 + x) 0

-- 7a
-- rmChar :: Char -> String -> String
-- rmChar c l = foldr ( \chr acc -> if chr==c then acc else chr : acc) [] l

-- -- 7b
-- rmCharsRec :: [String] -> String -> String
-- rmCharsRec _ [] = []  
-- rmCharsRec [] s = s  
-- rmCharsRec (c:cs) s = rmCharsRec cs (foldr rmChar c s)


-- lab 6 


-- data Fruct
--     = Mar String Bool
--     | Portocala String Int

-- ionatanFaraVierme = Mar "Ionatan" False
-- goldenCuVierme = Mar "Golden Delicious" True
-- portocalaSicilia10 = Portocala "Sanguinello" 10
-- cosFructe = [Mar "Ionatan" False,

--     Portocala "Sanguinello" 10,
--     Portocala "Valencia" 22,
--     Mar "Golden Delicious" True,
--     Portocala "Sanguinello" 15,
--     Portocala "Moro" 12,
--     Portocala "Tarocco" 3,
--     Portocala "Moro" 12,
--     Portocala "Valencia" 2,
--     Mar "Golden Delicious" False,
--     Mar "Golden" False,
--     Mar "Golden" True]

-- -- 1a
-- ePortocalaDeSicilia :: Fruct -> Bool 
-- ePortocalaDeSicilia (Portocala s _) = s `elem` ["Tarocco","Moro", "Sanguinello"]

-- -- 1b
-- nrFeliiSicilia :: [Fruct] -> Int
-- nrFeliiSicilia list= sum [ i |Portocala s i<- list, ePortocalaDeSicilia (Portocala s i)] 

-- -- 1c
-- nrMereViermi :: [Fruct] -> Int
-- nrMereViermi l = sum [ 1 | Mar _ t <- l, t == True ]

-- -- 2a
-- type NumeA = String
-- type Rasa = String
-- data Animal = Pisica NumeA | Caine NumeA Rasa
--     deriving Show

-- vorbeste :: Animal -> String
-- vorbeste (Pisica _) = "meow"
-- vorbeste (Caine _ _) = "Woof"

-- -- 2b
-- rasa :: Animal -> Maybe String
-- rasa (Pisica _) = Nothing
-- rasa (Caine _ rasa) = Just rasa

-- -- 3
-- data Linie = L [Int]
--     deriving Show
-- data Matrice = M [Linie]
--     deriving Show

-- -- 3a
-- verifica :: Matrice -> Int -> Bool
-- verifica (M linii) n = foldr (\(L l) acc -> acc && (sum l == n)) True linii

-- -- 3b
-- doarPozN :: Matrice -> Int -> Bool
-- doarPozN (M linii) n = verifica linii
--   where
--     verifica [] = True
--     verifica ((L l):rest)
--       | length l /= n = False
--       | not (all (>0) l) = False
--       | otherwise = verifica rest

      
-- pt main
-- let l = [0,1,-3,-2,8,-1,6]
    -- let n =5
    -- let m=10
    -- let result = listToInt [2,3,4,5]
    
    -- let listaFructe = [  Mar "Golden Delicious" True, Mar "Golden Delicious" False,Mar "Golden" False,Mar "Golden" True]
    -- let test_nrMereViermi = nrMereViermi listaFructe
    -- print test_nrMereViermi

    -- let cat = Pisica "Whiskers"
    -- let dog = Caine "Fido" "Labrador"

    -- print (rasa cat)
    -- print (rasa dog)

-- foo1 :: (Int,Char,String) -> String
-- foo1 (x, y, z) = "Result: " ++ show x ++ [y] ++ z

-- h x = x + g
--     where g x = x + 1

-- main :: IO ()
-- main = do
    
--     let f x = x + x 
--     let  g x = x * x
--     print (f 3)




-- ----------------------------------------------------------------------------RECAPITULARE ---------------------------------------------------------------------------------

-- ----------------------------------------------------------------------------LAB 2
-- 1
-- poly :: Double -> Double -> Double -> Double -> Double
-- poly a b c x = a * x^2 + b*x + c

-- -- 2
-- eeny :: Integer -> String
-- eeny x | even x = "eeny"
--        | otherwise = "meeny"

-- -- 3 garzi / conditii
-- fizzbuzz :: Integer -> String
-- fizzbuzz x | x `mod` 3 ==0 && x `mod` 5 == 0 = "fizzbuzz"
--            | x `mod` 3 == 0 = "Fizz"
--            | x `mod` 5 == 0 = "Buzz"
--            | otherwise =  " "

-- -- 4 recursivitate
-- tribonacci :: Integer -> Integer
-- tribonacci n 
--   | n == 1 || n == 2 = 1
--   | n == 3 = 2
--   | otherwise = tribonacci(n-1) + tribonacci(n-2) + tribonacci(n-3)

-- --5 coeficienti binomiali cu recursivitate
-- binomial :: Integer -> Integer -> Integer
-- binomial n k
--   | n == 0 = 0
--   | k == 0 = 1
--   | otherwise = binomial (n-1) k + binomial (n-1) (k-1)

-- --liste 
-- --6 a) 
-- verifL :: [Int] -> Bool
-- verifL l = even (length l) 

-- --6 b
-- takefinal ::[Int] -> Int -> [Int]
-- takefinal l n 
--   | length( l) >=n = drop (length l - n) l
--   | otherwise = l

-- -- 6 c
-- remove ::[Int] -> Int ->[Int]
-- remove l n = take (n-1) l ++ drop n l

-- --7 a
-- myreplicate :: Int -> Int ->[Int]
-- myreplicate n v 
--   | n <= 0 = []
--   | otherwise = v : myreplicate (n-1) v

-- --7 b
-- sumImp :: [Int] -> Int
-- sumImp [] =0
-- sumImp (h:t)
--   | odd h = h+ok
--   | otherwise = ok
--   where ok = sumImp t

-- --7c
-- totalLen :: [String] -> Int
-- totalLen [] = 0
-- totalLen (h:t)
--   | head h == 'A' = length h + ok
--   | otherwise = ok
--   where ok = totalLen t

-- main :: IO ()
-- main = do
    
--     let result = totalLen ["Ana"]
--     let r = binomial 0 2
--     let ri = binomial 5 6
--     let ry = tribonacci 12
  
--     print result
    -- print r
    -- print ri
    -- print ry

    
-- -----------------------------------------------------------------------------LAB 3

--1 liste si recursivitate : nr total de voc din sirurile palindrom
-- palindrom :: String -> Bool
-- palindrom s 
--   | s == reverse s = True
--   | otherwise = False

-- vocale :: String -> Int
-- vocale ""=0
-- vocale (i:f)
--   | i `elem` "aeiou" = ok +1
--   | otherwise = ok
--   where ok = vocale f

-- nrVocale :: [String] -> Int
-- nrVocale [] = 0
-- nrVocale (h:t) 
--   | palindrom h = vocale h + d
--   | otherwise = d
--   where d = nrVocale t

-- ----------------- liste definite cu proprietati caracteristice sau prin selectie
-- --3: lista div unui nr
-- divizori :: Int -> [Int]
-- divizori n = [ x | x <- [1..n], n `mod` x == 0 ]

-- --5
-- inInterval :: Int -> Int -> [Int] ->[Int]
-- inInterval x y l = [ n | n <- l, n>=x && n<=y]

-- --5 recursiv
-- inIntervalRec :: Int -> Int -> [Int] ->[Int]
-- inIntervalRec _ _ [] = []
-- inIntervalRec x y (h:t)
--   | h >= x && h <= y = h : inIntervalRec x y t
--   | otherwise = inIntervalRec x y t

-- --6 numara nr > 0
-- pozitive :: [Int] -> Int
-- pozitive [] = 0
-- pozitive (h:t)
--   | h > 0 = ok + 1
--   | otherwise = pozitive t
--   where ok = pozitive t



-- main :: IO ()
-- main = do
    
--     let result = pozitive [0,1,-3,-2,8,-1,6]
--     --let  g x = x * x
--     print (result )


-- --------------------------------------------------------------------------------LAB 4

--2 lista divizorilor pozitivi ai unui nr
-- factori :: Int -> [Int]
-- factori n = [ x | x <- [ 1..n], n `mod` x == 0]

-- --3 folosinf factori, defineste predicatul prim car everif daca un nr primit ca parametru e prim
-- prim :: Int -> Bool
-- prim m = length ( factori m ) ==2
  
-- --4 pt un numar n primit ca param, intoarce lista nr prime din intervalul [2..n]
-- numerePrime :: Int -> [Int]
-- numerePrime n = [ x | x <- [2..n], prim x]

-- ---------------------------map si filter

-- --6
-- firstEl :: [(a,b)] -> [a]
-- firstEl p = map (\(x, _) -> x) p

-- -- sau

-- --firstEl ls = map fst ls         -- fst scoate primul element dintr-un tuplu

-- --7
-- sumList :: [[Int]] -> [Int]
-- sumList l = map sum l

-- --8
-- prel2 :: [Int] -> [Int]
-- prel2 l = map ( \x -> if even x then x `div` 2 else x*2) l

-- --9
-- foo :: Char -> [String] -> [String]
-- foo c l = filter ( \x -> c `elem` x ) l
-- --sau foo c ls = filter (elem c) ls    -- aplicam 'elem c' pe sirurile noastre si returneaza true daca caracterul se afla in el

-- --10
-- f :: [Int] ->[Int]
-- f l = filter odd ( map (^2) l) 

-- --11 
-- f1 :: [Int] -> [Int]
-- f1 l =  map( (^2) . snd ) $ filter (odd . fst) $ zip [1..] l

-- main :: IO ()
-- main = do
    
--     let result = f1 [ 1,2,3,4,5]
--     --let  g x = x * x
--     print (result )

-- --------------------------------------------------------------------------------LAB 5

----------------------------FOLD
-- map filter si fold 

--3 verifica daca TOATE elementele..
-- allVerifies :: (Int -> Bool) -> [Int] -> Bool
-- allVerifies prop l= foldr (\a b -> prop a && b) True l

-- --4 verifica daca EXISTA elemente care satisfac
-- anyVerifies :: (Int -> Bool) -> [Int] -> Bool
-- anyVerifies prop l = foldr (\a b -> prop a || b) False l

-- --6
-- listToInt :: [Integer] -> Integer
-- listToInt l = foldl (\a b -> 10*a +b) 0 l

-- --7
-- rmChar :: Char -> String -> String
-- rmChar c s = foldr ( \a b -> if a/= c then a:b else b) "" s

-- --8 
-- myReverse :: [Int] -> [Int]
-- myReverse l = foldl ( \a b -> b:a) [] l


-- main :: IO ()
-- main = do
    
--     let result = myReverse [1,2,3]
--     -- let  g x = x * x
--     print (result)
-- -------------------------------------------LAB 6
---------------------------------------------------------TIPURI DE DATE

-- data Fruct
--   = Mar String Bool
--   | Portocala String Int

-- -- ionatanFaraVierme = Mar "Ionatan" False
-- -- goldenCuVierme = Mar "Golden Delicious" True
-- -- portocalaSicilia10 = Portocala "Sanguinello" 10

-- cosFructe = [Mar "Ionatan" False,
--                 Portocala "Sanguinello" 10,
--                 Portocala "Valencia" 22,
--                 Mar "Golden Delicious" True,
--                 Portocala "Sanguinello" 15,
--                 Portocala "Moro" 12,
--                 Portocala "Tarocco" 3,
--                 Portocala "Moro" 12,
--                 Portocala "Valencia" 2,
--                 Mar "Golden Delicious" False,
--                 Mar "Golden" False,
--                 Mar "Golden" True]


-- --1 a 
-- ePortocalaDeSicilia :: Fruct -> Bool
-- ePortocalaDeSicilia (Portocala tip _) = tip `elem` [ "Tarocco", "Moro", "Sanguinello"] 
-- ePortocalaDeSicilia _= False

-- --1b
-- nrFeliiSicilia :: [Fruct] -> Int
-- nrFeliiSicilia l =sum [ i | Portocala s i <- l, ePortocalaDeSicilia(Portocala s i) ]

-- --1c
-- nrMereViermi :: [Fruct] -> Int
-- nrMereViermi l = sum [ 1 | Mar tip True <-l]

-- --2 -TIPUL DE DATE MAYBE
-- type NumeA = String
-- type Rasa = String
-- data Animal = Pisica NumeA | Caine NumeA Rasa
--   deriving Show

-- --2a
-- vorbeste :: Animal -> String
-- vorbeste (Pisica _ ) = "Miau"
-- vorbeste (Caine _ _) =" Woof"
-- --2b
-- rasa :: Animal -> Maybe String
-- rasa (Pisica _) = Nothing
-- rasa (Caine _ r) = Just (r)

-- --3 Matrix Resurrections
-- data Linie = L [Int]
--   deriving Show
-- data Matrice = M [Linie]
--   deriving Show

-- --3a
-- -- matrice1 = (M[L[1,2,3], L[4,5], L[2,3,6,8], L[8,5,3]])
-- -- matrice2 = (M[L[2,20,3], L[4,21], L[2,3,6,8,6], L[8,5,3,9]])
-- verifica :: Matrice -> Int -> Bool
-- verifica (M linii) n = foldr ( \(L a) b -> n == sum a && b) True linii

-- --3b
-- doarPozN :: Matrice -> Int -> Bool
-- doarPozN (M linii) n = foldr ( \(L a) b -> if length a == n then all(>0) a && b else b ) True linii

-- --3c
-- corect :: Matrice -> Bool
-- corect (M []) =True
-- corect (M ((L l) : ls)) = foldr(\(L a) b -> length a == length ls && b) True ls

-- main :: IO ()
-- main = do
    
--     let test_ePortocalaDeSicilia1 = ePortocalaDeSicilia (Portocala "Moro" 12)
--     let  test_ePortocalaDeSicilia2 = ePortocalaDeSicilia (Mar "Ionatan" True)
--     let test_nrFeliiSicilia = nrFeliiSicilia cosFructe
--     let test_nrMereViermi = nrMereViermi cosFructe
--     let testPisica = rasa (Pisica "Sasha")
--     let testCaine = rasa (Caine "Charlie" "Ciobanesc")
--     let test_verif2 = verifica (M[L[2,20,3], L[4,21], L[2,3,6,8,6], L[8,5,3,9]]) 25
--     let test_verif1 = verifica (M[L[1,2,3], L[4,5], L[2,3,6,8], L[8,5,3]]) 10
--     let testPoz1 = doarPozN (M [L[1,2,3], L[4,5], L[2,3,6,8], L[8,5,3]]) 3
--     let testcorect1 = corect (M[L[1,2,3], L[4,5], L[2,3,6,8], L[8,5,3]])
--     print (testcorect1)




-- ----------------LAB 7

---------------------------------EXPRESII SI ARBORI
-- data Expr = Const Int -- integer constant
--   | Expr :+: Expr -- addition
--   | Expr :*: Expr -- multiplication
--   deriving Eq

-- data Operation = Add | Mult deriving (Eq, Show)
-- data Tree = Lf Int -- leaf
--   | Node Operation Tree Tree -- branch
--   deriving (Eq, Show)

-- instance Show Expr where
--   show (Const x) = show x
--   show (e1 :+: e2) = "(" ++ show e1 ++ " + "++ show e2 ++ ")"
--   show (e1 :*: e2) = "(" ++ show e1 ++ " * "++ show e2 ++ ")"


-- --1 
-- --exemple
-- exp1 = ((Const 2 :*: Const 3) :+: (Const 0 :*: Const 5))
-- exp2 = (Const 2 :*: (Const 3 :+: Const 4))
-- exp3 = (Const 4 :+: (Const 3 :*: Const 3))
-- exp4 = (((Const 1 :*: Const 2) :*: (Const 3 :+: Const 1)) :*: Const 2)
-- test11 = evalExp exp1 
-- test12 = evalExp exp2 
-- test13 = evalExp exp3 
-- test14 = evalExp exp4 

-- evalExp :: Expr -> Int
-- evalExp (Const i) = i
-- evalExp( e :+: e2) =evalExp e+ evalExp e2
-- evalExp (e1 :*: e2) = evalExp e1 * evalExp e2

-- --2
-- --exemple
-- arb1 = Node Add (Node Mult (Lf 2) (Lf 3)) (Node Mult (Lf 0)(Lf 5))
-- arb2 = Node Mult (Lf 2) (Node Add (Lf 3)(Lf 4))
-- arb3 = Node Add (Lf 4) (Node Mult (Lf 3)(Lf 3))
-- arb4 = Node Mult (Node Mult (Node Mult (Lf 1) (Lf 2)) (Node Add (Lf 3)(Lf 1))) (Lf 2)
-- test21 = evalArb arb1 
-- test22 = evalArb arb2 
-- test23 = evalArb arb3 
-- test24 = evalArb arb4 

-- evalArb :: Tree -> Int
-- evalArb (Lf val) = val
-- evalArb (Node Add t1 t2) = evalArb t1 + evalArb t2
-- evalArb ( Node Mult t1 t2) = evalArb t1 * evalArb t2

-- --3
-- expToArb :: Expr -> Tree
-- expToArb (Const v) = Lf v
-- expToArb ( e1 :+: e2) = Node Add (expToArb e1) (expToArb e2)
-- expToArb ( e1 :*: e2) = Node Mult (expToArb e1) (expToArb e2)


-- -------------------------------------------------ARBORI BINARI DE CAUTARE
-- data IntSearchTree value
--     = Empty
--     | BNode
--         (IntSearchTree value) -- elemente cu cheia mai mica
--         Int -- cheia elementului
--         (Maybe value) -- valoarea elementului
--         (IntSearchTree value) -- elemente cu cheia mai mare

-- --4 --cauta un elem in arbore
-- lookup' :: Int -> IntSearchTree value -> Maybe value
-- lookup' _ Empty = Nothing
-- lookup' key (BNode right actualKey value left)
--   |key == actualKey = value
--   |key < actualKey = lookup' key left
--   |otherwise = lookup' key right

-- --5 lista cheilor nodurilor
-- keys :: IntSearchTree value -> [Int]
-- keys Empty = []
-- keys (BNode right actualKey value left) = keys left ++ [actualKey] ++ keys right

-- --6 lista valorilor nodurilor
-- values :: IntSearchTree value -> [value]
-- values Empty = []
-- values (BNode left _ (Just value) right) = values left ++ [value] ++ values right
-- values (BNode left _ Nothing right) = values left ++ values right



-- main :: IO ()
-- main = do
    
   
--     let  g x = x * x
--     print (test23)
-- ----------------LAB 8
---------------------------------------CLASE DE TIPURI

-- class Collection c where
--   empty :: c key value                                      --crearea unei colectii vide
--   singleton :: key -> value -> c key value                  --crearea unei colectii cu un element
--   insert :: Ord key => key -> value -> c key value -> c key value -- adaugarea/actualizarea unui element intr-o colectie       
--   clookup :: Ord key => key -> c key value -> Maybe value   --cautarea unui element intr-o colectie
--   delete :: Ord key => key -> c key value -> c key value    --stergerea (marcarea ca sters a) unui element dintr-o colectie
--   keys :: c key value -> [key]                              --obtinerea listei cheilor
--   values :: c key value -> [value]                          --obtinerea listei valorilor
--   toList :: c key value -> [(key, value)]                   --obtinerea listei elementelor
--   fromList :: Ord key => [(key,value)] -> c key value 

-- --1 
--   keys a = map (fst) (toList a)
--   values a = map (snd) (toList a)
--   fromList [] = empty
--   fromList ((key, value) : xs) = insert key value (fromList xs)  -- adaugam elementul unu cate unu folosind insertt

-- --2  Definiti o instanta pentru Pairlist a clasei collection
-- newtype PairList k v
--   = PairList { getPairList :: [(k, v)] }

-- instance Collection PairList where
--     empty :: PairList a b
--     empty = PairList []  -- contructor de date + lista vida

--     singleton :: key ->value -> PairList key value
--     singleton key value = PairList[(key, value)]

--     insert ::  Ord key => key -> value -> PairList key value -> PairList key value
--     insert key value (PairList ls) =  PairList $ (key, value) : ls

--     clookup :: Ord key => key -> PairList key value -> Maybe value
--     clookup _ (PairList []) = Nothing
--     clookup key (PairList ls) = if not $ null val then Just . snd $ head val else Nothing   -- daca lista nu e vida atunci returnam Just valoare, altfel nothing
--                                 where val = [(key, value) | (k, value) <- ls, k == key] --contruim lista care contine elemente care au cheia data

--     delete :: Ord key => key -> PairList key value -> PairList key value
--     delete key (PairList ls) = PairList [(k, value) | (k, value) <- ls, k /= key]   -- generez lista fara elementele cu cheia data

--     toList = getPairList    -- folosim functia care ia lista din PairList, este data in definitie

-- --
-- data Punct = Pt [Int]

-- data Arb = Vid | F Int | N Arb Arb
--   deriving Show

-- class ToFromArb a where
--     toArb :: a -> Arb
--     fromArb :: Arb -> a
-- -- instanta a clasei show  pentru tipul de date Punct, astfel incat lista coordonatelor sa fie afisata ca tuplu

-- -- instance Show Punct where
-- --     show :: Punct -> String
-- --     show Pt([]) = "()"
-- --     show (Pt ls) = "(" ++ showSep lx ++ ")"
-- --                     where showSep [x] = show ls
-- --                           showSep [x: xs] = show x ++ "," ++ showSep xs

-- -- instanta a clasei ToFromArb pt tipul de date Punct, ai coordonatele pct sa coincida cu frontiera arborelui

-- instance ToFromArb Punct where
--     toArb :: Punct -> Arb
--     toArb ( Pt []) = Vid
--     toArb (Pt (x:xs)) = N ( F x) $ toArb (Pt xs)

--     fromArb :: Arb ->Punct
--     fromArb (Vid) = Pt []
--     fromArb ( F a) = Pt [a]
--     fromArb (N c d) = Pt $ m ++ n
--                         where Pt m = fromArb c
--                               Pt n = fromArb d 

-- --figuri geometrice
-- data Geo a = Square a | Rectangle a a | Circle a
--   deriving Show

-- class GeoOps g where
--   perimeter :: (Floating a) => g a -> a
--   area :: (Floating a) => g a -> a

-- --instantiati clasa GeoOps pt tipul de date Geo 

-- instance GeoOps Geo where
--     perimeter :: (Floating a) => Geo a-> a
--     perimeter (Square a) = 4*a
--     perimeter ( Rectangle a b) = 2*a+2*b
--     perimeter(Circle a) = 2*pi*a

--     area :: (Floating a) => Geo a -> a
--     area (Square a) = a*a
--     area ( Rectangle a b) = a*b
--     area (Circle a) = pi * a*a

-- --instantiere clasa Eq pt tipul Geo, ai 2 fig sa fie egale daca au perim egal

-- instance (Eq a, Floating a ) => Eq (Geo a ) where
--     (==) :: Geo a -> Geo a -> Bool
--     g1 == g2 = perimeter g1 == perimeter g2 

    

-- main :: IO ()
-- main = do
    
--     let f x = x + x 
--     let  g x = x * x
--     print (f 3)
-- ----------------LAB 9
-------------------------------------CLASE DE TIPURI CONTINUARE

data Tree = Empty -- arbore vid
  | Node Int Tree Tree Tree -- arbore cu valoare de tip Int in radacina


class ArbInfo t where
  level :: t -> Int -- intoarce inaltimea arborelui;
-- consideram ca un arbore vid are inaltimea 0
  sumval :: t -> Int -- intoarce suma valorilor din arbore
  nrFrunze :: t -> Int -- intoarce nr de frunze al arborelui


instance ArbInfo Tree where
    level :: Tree -> Int
    level Empty = 0
    level (Node _ t1 t2 t3 ) =  1 + max (max (level t1)  (level t2)) (level t3)

    sumval :: Tree -> Int
    sumval (Node val t1 t2 t3) = val + sumval t1 + sumval t2 + sumval t3
    nrFrunze :: Tree -> Int
    nrFrunze (Node _ Empty Empty Empty) = 1
    nrFrunze (Node _ t1 t2 t3) = nrFrunze t1 + nrFrunze t2 + nrFrunze t3

extree :: Tree
extree = Node 4 (Node 5 Empty Empty Empty)
  (Node 3 Empty Empty (Node 1 Empty Empty Empty)) Empty


---------------------------------------------------------------------------------------------------
data Prop = V String | T | F | Prop :&: Prop | Prop :|: Prop
  deriving (Show, Eq)

class Operations exp where
  simplify :: exp -> exp

--instanta a clasei operations pt tipul de date prop, ai op de simplif sa efect calc bool cu T si F

instance Operations Prop where
  simplify :: Prop -> Prop 
  simplify ( V s) = V s
  simplify (T) = T
  simplify (F) = F 
  simplify (p :&: T) = simplify(p)
  simplify ( T :&: p) = simplify (p)
  simplify (F :&: p) = simplify (F)
  simplify ( p :&: F) = simplify(F)
  simplify (p :|: T) = T
  simplify (T :|: p) = T
  simplify (p :|: F) = simplify(p)
  simplify (F :|: p) = simplify(p)
  simplify ( p :&: q) = simplify p :&: simplify q
  simplify ( p :|: q) = simplify p :|: simplify q

prop1 = ((V "p") :|: (V "q")) :&: T
prop2 = prop1 :|: (V "r")
prop3 = ((F :&: V "p") :|: (V "q"))
prop4 = prop3 :&: (V "q")

main :: IO ()
main = do
    
    --let result = getFrominterval 5 7 [1..10]
    --let  g x = x * x
    print (prop1)
-- ----------------LAB 10

-- main :: IO ()
-- main = do
    
--     let f x = x + x 
--     let  g x = x * x
--     print (f 3)
-- ----------------LAB 11

-- main :: IO ()
-- main = do
    
--     let f x = x + x 
--     let  g x = x * x
--     print (f 3)
-- ----------------LAB 12

-- main :: IO ()
-- main = do
    
--     let f x = x + x 
--     let  g x = x * x
--     print (f 3)
-- ----------------LAB 13

-- main :: IO ()
-- main = do
    
--     let f x = x + x 
--     let  g x = x * x
--     print (f 3)
