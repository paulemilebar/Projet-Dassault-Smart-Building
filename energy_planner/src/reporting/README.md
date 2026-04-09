# Reporting Dashboard With LLM README

Ce README a pour but de décrypter les scripts de reporting qui permettent de générer un dashboard 24h du smart building, puis d'y ajouter un résumé en langage naturel produit par un LLM via OpenRouter.

## Fichiers principaux

- `dashboard_24h_with_vizualization.py`
  Script principal pour générer le dashboard 24h avec visualisation et resumé en langage naturel

- `optimization_summary.py`
  Construit le payload transmis au LLM a partir des résultats d'optimisation, des métriques energétiques et de la comparaison baseline et l'optimiseur.

- `dispatch_visualization_parameters.py`
  Fichier de configuration local a créer (si ce n'est pas déjà fait) ou compléter pour stocker la cle API OpenRouter. A NOTER : Ce fichier apparait dans le git ignore CAR IL NE FAUT PAS POUSSER SUR LE GIT REPO VOTRE CLE API. ELLE EST SECRETE ET PERSONNELLE.

## Comment le dashboard avec LLM fonctionne

Le pipeline suit les étapes suivantes :

1. Chargement des données prédites via `load_predicted_inputs`.
2. Chargement de l'état du système via `load_current_state`.
3. Exécution de l'optimiseur pour produire le plan horaire (`Pin`, `Pgo`, `PV`, `Pch`, `Pdis`, `Ebat`, `S`).
4. Construction du dashboard Plotly avec les courbes et indicateurs.
5. Calcul d'une comparaison entre :
   - une baseline simple qui ne ferait qu'acheter au main grid pour répondre à la demande utilisateur,
   - le résultat réel de l'optimiseur.
6. Construction d'un payload de synthèse pour le LLM :
   - énergies achetées/vendues,
   - utilisation PV,
   - charge/décharge batterie,
   - cout net,
   - emissions,
   - comparaison baseline,
   - formulation du modèle d'optimisation (objectif + contraintes principales).
7. Envoi de ce payload a OpenRouter pour générer un resumé en langage naturel en francais
8. Insertion de ce résumé a la fin du dashboard HTML

Le LLM ne "voit" donc pas directement les graphes : il recoit un résume structuré des résultats numériques et du modèle d'optimisation, puis rédige une explication naturelle fidèle aux chiffres.

## Fichier de configuration a creer

Pour que le dashboard avec LLM fonctionne, il faut disposer d'un fichier :

`energy_planner/src/reporting/dispatch_visualization_parameters.py`

Ce fichier doit exposer la variable suivante :

```python
OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx"
```

Sans ce fichier, ou si la clé est absente, le script ne pourra pas interroger OpenRouter et donc ne pourra PAS de connecter au LLM.

## Choix du modele OpenRouter

Le modèle utilisé peut changer selon :

- la disponibilité du modèle sur OpenRouter,
- sa gratuite ou non au moment de l'execution,
- les limitations temporaires de l'API

En pratique, le choix du modèle depend donc de ce qui est accessible gratuitement a l'instant t sur OpenRouter d'où le fait qu'on a pas de modèle pré-établis.

De maniere générale, le modèle utilisé est :

`Nemotron 3 Super`

Mais ce choix n'est pas garanti en permanence. Si ce modèle n'est pas disponible ou n'est plus gratuit, il faut remplacer la valeur du modele dans le script par un autre modele compatible OpenRouter

## Point important

La cle API OpenRouter est une configuration locale. Il ne faut pas committer une vraie clé secrète sur le répo du GIT.
