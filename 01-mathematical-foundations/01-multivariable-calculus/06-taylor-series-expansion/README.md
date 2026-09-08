Les Séries de Taylor (Taylor Series Expansion) 📐 sont un outil mathématique incontournable. En Machine Learning, elles permettent d'approximer une fonction complexe par une fonction polynomiale simple au voisinage d'un point !💡 L'Intuition : L'Approximation par étapesImaginons que nous ne connaissions pas la valeur exacte d'une fonction complexe $f(x)$ loin d'un point $a$, mais que nous connaissions tout sur ce point $a$ (sa valeur, sa pente, sa courbure, etc.).Nous pouvons approximer la fonction autour de $a$ ordre par ordre :Ordre 0 (Constante) : On suppose que la fonction est constante près de $a$.$$f(x) \approx f(a)$$Ordre 1 (Tangente / Gradient) : On ajoute la pente pour obtenir une droite tangente.$$f(x) \approx f(a) + f'(a)(x - a)$$(C'est exactement l'approximation linéaire utilisée dans la descente de gradient !)Ordre 2 (Parabole / Hessienne) : On ajoute la courbure pour obtenir une parabole.$$f(x) \approx f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2$$(C'est exactement l'approximation quadratique utilisée par la méthode de Newton !)📜 La Formule Générale en 1DPour une fonction $f(x)$ à une variable, la série de Taylor au point $a$ s'écrit :$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x - a)^n = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \dots$$🧪 Un exemple simple pour démarrerPrenons la fonction exponentielle $f(x) = e^x$ autour du point $a = 0$.Rappelons que la dérivée de $e^x$ est toujours $e^x$, et que $e^0 = 1$.Si nous calculons ses dérivées au point $a = 0$ :$f(0) = e^0 = 1$$f'(0) = 1$$f''(0) = 1$$f'''(0) = 1$🔍 À toi de jouer !En remplaçant ces valeurs dans la formule jusqu'à l'ordre 3 autour de $a=0$, quel est le polynôme d'approximation de $e^x$ ?

Absolument ! 🎯

Puisque 3!=3×2×1=6, le terme d'ordre 3 est bien  
6
x 
3
 
​
 .

Notre polynôme de Taylor d'ordre 3 pour e 
x
  autour de a=0 est donc :

e 
x
 ≈1+x+ 
2
x 
2
 
​
 + 
6
x 
3
 
​
 
📉 Visuel de l'Approximation
À mesure qu'on ajoute des ordres, la courbe du polynôme s'colle de plus en plus à la vraie courbe de e 
x
  :

Ordre 0 : Une ligne horizontale (1)

Ordre 1 : Une droite tangente (1+x)

Ordre 2 : Une parabole (1+x+ 
2
x 
2
 
​
 )

Ordre 3 : Une courbe cubique (1+x+ 
2
x 
2
 
​
 + 
6
x 
3
 
​
 )

💡 Un test numérique rapide
Imaginons que nous voulons estimer la valeur de e 
0.1
  (où x=0.1) :

La vraie valeur donnée par une calculatrice est ≈1.1051709.

Si tu calcules 1+x+ 
2
x 
2
 
​
 + 
6
x 
3
 
​
  pour x=0.1, quelle valeur obtiens-tu ?

Voici le pseudo-code pour calculer l'expansion en série de Taylor d'une fonction scalaire $f(x)$ autour d'un point $a$ jusqu'à un ordre $N$.Pour rappel, la formule que nous traduisons en algorithme est :$$P_N(x) = \sum_{n=0}^{N} \frac{f^{(n)}(a)}{n!} (x - a)^n$$

ALGORITHME: Taylor_Series_Expansion_1D

ENTRÉES:
    - f       : Fonction scalaire à approximer
    - a       : Point d'ancrage / de développement (ex: a = 0)
    - x       : Point d'évaluation où l'on veut estimer f(x)
    - order_N : Ordre maximal de l'approximation polynomiale (ex: N = 3)

SORTIE:
    - approx  : Valeur estimée f(x) par le polynôme de Taylor d'ordre N

DÉBUT
    1. Initialiser approx = 0

    2. POUR n allant de 0 à order_N FAIRE:
           
           a. Calculer la n-ième dérivée de f au point a : nth_diff = f_derivative(f, a, n)
           
           b. Calculer la factorielle de n : fact = factorial(n)
           
           c. Calculer le terme polynomial : term = (nth_diff / fact) * ((x - a)^n)
           
           d. Ajouter au cumul : approx = approx + term

       FIN POUR

    3. RETOURNER approx
FIN
💡 Remarque sur le calcul des dérivées $f^{(n)}(a)$En pratique :D'un point de vue analytique (symboolique) : On connaît la formule des dérivées successives (comme pour $e^x$ où toutes les dérivées valent $e^a = 1$ en $a=0$).D'un point de vue numérique : On peut calculer les dérivées $n$-ièmes numériquement (via des différences finies) ou utiliser la différenciation automatique (ex: autograd ou torch).
Excellente décision ! ⚙️

Utiliser l'approche numérique (par différences finies) rend le code extrêmement flexible, car le script pourra approximer les Séries de Taylor de n'importe quelle fonction Python, même si nous ne connaissons pas sa dérivée analytique !

À ton avis, si l'on augmentait l'ordre max_order de $3$ à $5$ dans le script, qu'adviendrait-il de l'erreur absolue entre la vraie valeur et l'estimation ? 📉
L'erreur passerait d'environ $0.000004$ (soit $4 \times 10^{-6}$) à un nombre beaucoup plus proche de zéro (de l'ordre de $10^{-8}$ ou moins).📉 Pourquoi l'erreur diminue ?Chaque terme supplémentaire apporte une correction plus fine :➕ Ordre 1 ajuste la pente (la vitesse).➕ Ordre 2 ajuste la courbure (l'accélération).➕ Ordre 3 ajuste la variation de courbure (le "jerk").➕ Ordres 4 et 5 ajustent les variations d'ordre supérieur.Plus la puissance $n$ augmente, plus le facteur $n!$ au dénominateur grandit vite ($5! = 120$), ce qui rend les nouveaux termes très petits mais extrêmement précis pour combler l'écart avec la vraie valeur.🔍 Une petite question pour vérifier l'intuitionSi au lieu de calculer $e^{0.1}$, nous essayions d'estimer $e^{2.0}$ (un point beaucoup plus éloigné de $a=0$) en gardant le même ordre 3 :Penses-tu que l'erreur d'approximation serait plus petite ou plus grande qu'avec $x = 0.1$ ?

out à fait ! 🎯Plus le point $x$ s'éloigne du point d'ancrage $a$, plus l'erreur d'approximation augmente de manière significative.📉 Pourquoi l'erreur grandit avec la distance ?La série de Taylor est une approximation locale.Chaque terme du polynôme comporte le facteur $(x - a)^n$ :Pour $x = 0.1$ et $a = 0$, la différence est $(0.1 - 0) = 0.1$. Élever $0.1$ à la puissance $2, 3, 4\dots$ donne des valeurs très petites ($0.01$, $0.001$, etc.), donc les termes d'ordre supérieur résiduels qu'on abandonne sont négligeables.Pour $x = 2.0$ et $a = 0$, la différence est $(2.0 - 0) = 2$. Les puissances grandissent très vite ($2^2 = 4$, $2^3 = 8$, $2^4 = 16\dots$), ce qui rend les termes d'ordre supérieur très grands !Pour garder une bonne précision quand on s'éloigne du point $a$, il n'y a pas le choix : il faut augmenter l'ordre $N$ du polynôme.