module ModeleExamene where
import Data.Binary.Get (label)
import GHC.Lexeme (okSymChar)
import Control.Applicative (Applicative(liftA2))
import Distribution.Simple.Setup (trueArg)
import Distribution.Compat.Exception (tryIO)
import Control.Arrow (ArrowChoice(left))
import Text.XHtml (base, abbr, name, background)
import Text.Parsec.Language (mondrianDef)
import Graphics.Win32 (vK_OEM_1)
import Data.Char
import Data.Time.Calendar.MonthDay (monthAndDayToDayOfYear)
import Data.Time.Calendar.OrdinalDate (mondayStartWeek)
import System.Win32 (COORD(xPos))
import GHC.Platform.ArchOS (PPC_64ABI(ELF_V2))


-- 2024 Model

-- data Point = Pt [Int]
--   deriving Show

-- data Arb = Empty | Node Int Arb Arb
--   deriving Show

-- class ToFromArb a where
--   toArb :: a -> Arb
--   fromArb :: Arb -> a

-- --o instanta a clasei ToFormArb pt tipul Point

-- instance ToFromArb Point where

--     toArb ::Point -> Arb
--     toArb (Pt[]) = Empty
--     toArb ( Pt ls) = foldl (flip insert ) Empty ls
--                     where 
--                         insert x Empty = Node x Empty Empty   --daca ajunge sa compare x cu empty, se insereaza
--                         insert x ( Node val arb1 arb2) = if x<val
--                                                             then Node val (insert x arb1) arb2
--                                                         else 
--                                                             Node val arb1 (insert x arb2)

--     fromArb :: Arb -> Point
--     fromArb (Empty) = Pt ([])
--     fromArb ( Node i arb1 arb2) = Pt $ i: m ++ n
--                                   where Pt m = fromArb arb1
--                                         Pt n = fromArb arb2


--2
--fara monade
-- getFromInterval :: Int -> Int -> [Int] -> [Int]
-- getFromInterval a b l = [i| i<-l, i>=a && i<=b]

-- --cu monade
-- getFrominterval :: Int -> Int -> [Int] -> [Int]
-- getFrominterval a b l = do 
--     xl <- l 
--     if xl >= a && xl <=b then
--         do [xl]
--     else 
--         do []

-- --seria 24 v1
-- data Prop = V String | T | F | Prop :&: Prop | Prop :|: Prop
--   deriving (Show, Eq)

-- class Operations exp where
--   simplify :: exp -> exp

-- --instanta a clasei operations pt tipul de date prop, ai op de simplif sa efect calc bool cu T si F
-- ----NU E CORECT
-- -- instance Operations Prop where
-- --   simplify :: Prop -> Prop 
-- --   simplify ( s) = s  
-- --   simplify (p :&: T) = simplify p
-- --   simplify ( T :&: p) = simplify p
-- --   simplify (F :&: _) = F
-- --   simplify (_ :&: F) = F
-- --   simplify (p :|: T) = T
-- --   simplify (T :|: _) = T
-- --   simplify (p :|: F) = simplify p
-- --   simplify (F :|: p) = simplify(p)
-- --   simplify ( p :&: q) = simplify p :&: simplify q
-- --   simplify ( p :|: q) = simplify p :|: simplify q


-- ------------------------------------------------CORECT
-- instance Operations Prop where 
--     simplify (p1 :&: p2) = simplifyAnd (simplify p1) (simplify p2)
--     simplify (p1 :|: p2) = simplifyOr (simplify p1) (simplify p2)
--     simplify p = p 

-- simplifyAnd :: Prop -> Prop -> Prop 
-- simplifyAnd T p = p
-- simplifyAnd p T = p
-- simplifyAnd _ F = F
-- simplifyAnd F _ = F
-- simplifyAnd p1 p2 = p1 :&: p2 

-- simplifyOr :: Prop -> Prop -> Prop 
-- simplifyOr T _ = T
-- simplifyOr _ T = T
-- simplifyOr p F = p
-- simplifyOr F p = p
-- simplifyOr p1 p2 = p1 :|: p2

-- prop1 = ((V "p") :|: (V "q")) :&: T
-- prop2 = prop1 :|: (V "r")
-- prop3 = ((F :&: V "p") :|: (V "q"))
-- prop4 = prop3 :&: (V "q")

-- --2 limba pasareasca

-- sir :: String -> String
-- sir ""=""
-- sir (x:xs)
--   | isUpper x = (toLower x) : sir xs
--   | isLower x = toUpper x : sir xs
--   | isDigit x = '*' : sir xs
--   | otherwise = sir xs

-- --monada
-- sirM :: String -> String
-- sirM xs = do
--            x <- xs
--            if isUpper x
--             then do return ( toLower x)
--            else if isLower x
--             then do return(toUpper x)
--            else if isDigit x
--             then do return $ '*'
--            else ""
---alta varianta
-- sirMonad :: String -> String
-- sirMonad xs = do
--            x <- xs
--            if isUpper x
--             then toLower x : []
--            else if isLower x
--             then toUpper x : []
--            else if isDigit x
--             then ['*']
--            else ""

----------SERIA 23 v2

--1
data Expr = Var String | Val Int | Plus Expr Expr | Mult Expr Expr
  deriving (Show, Eq)

class Operations exp where
  simplify :: exp -> exp

instance Operations Expr where
  simplify :: Expr -> Expr
  simplify (Plus e (Val 0)) = simplify e
  simplify (Plus e (Var a)) = simplify e
  simplify (Mult (Val 0) _) = Val 0
  simplify (Mult _ (Val 0)) = Val 0
  simplify (Mult (Val 1) e) = simplify e
  simplify (Mult e (Val 1) ) = simplify e
  simplify (Plus e1 e2) = Plus (simplify e1) (simplify e2)
  simplify ( Mult e1 e2) = Mult (simplify e1) (simplify e2)
  simplify e = e


ex1 = Mult (Plus (Val 1) (Var "x")) (Val 1)
ex2 = Plus ex1 (Val 3)
ex3 = Plus (Mult (Val 0) (Val 2)) (Val 3)
ex4 = Mult ex3 (Val 5)

--2
sir :: String -> String
sir "" = ""
sir (x:xs)
  |x `notElem` "aeiouAEIOU" && isAlpha x= x : 'P' : x : sir xs
  |x `elem` "aeiouAEIOU" && isAlpha x= x: sir xs
  | otherwise = x: sir xs

--monade
sirM :: String -> String
sirM xs = do
           x <-xs
           if x `notElem` "aeiouAEIOU" && isAlpha x 
            then x : 'P' : [x]
           else if x `elem` "aeiouAEIOU" && isAlpha x 
            then [x]
           else ""




main :: IO ()
main = do
    
    let result = sirMonad "Ana,2"
    --let  g x = x * x
    print (result)
