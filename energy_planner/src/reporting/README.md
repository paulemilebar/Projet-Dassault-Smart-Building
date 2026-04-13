# Reporting Dashboard With LLM README
Ce README explique les scripts qui generent le dashboard 24h du smart building et le resume en langage naturel base sur un LLM.
## Fichiers principaux
- energy_planner/src/reporting/dashboard_24h_with_vizualization.py
  Script de reporting principal. Il execute l'optimiseur, calcule une baseline de comparaison, construit le dashboard Plotly et ajoute un resume LLM dans un rapport HTML.
- utils/dispatch_visualization.py
  Variante utilitaire en ligne de commande. Elle genere egalement un dashboard HTML et un resume LLM, avec passage possible des parametres via arguments
- energy_planner/src/reporting/optimization_summary.py
  Module central pour le résumé du LLM. Il construit le payload transmis au modele, cree le prompt et gere le fallback deterministe si l'appel LLM échoue.
- energy_planner/src/reporting/dispatch_visualization_parameters.py
  Fichier local a créer pour stocker la cle API OpenRouter et quiconque veut run le dashboard. Ce fichier est dans le gitignore et ne doit jamais contenir une cle API committée !

## Configuration locale attendue
Créer le fichier suivant :
energy_planner/src/reporting/dispatch_visualization_parameters.py
Contenu minimal :
`python
OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx"
`
La cle est personnelle et doit rester locale !

## Quel endpoint et quel SDK sont utilises
Le projet utilise le SDK Python openai, mais pas uniquement avec l'API OpenAI officielle.
Le module 	ry_generate_llm_summary(...) fonctionne avec tout endpoint compatible OpenAI.
Dans la configuration actuelle du projet, les valeurs par defaut sont :
- model = openrouter/free
- ase_url = https://openrouter.ai/api/v1
ATTENTION : le choix du modèle qu'on utilise n'est pas fixe. Il peut etre modifié :
- dans dashboard_24h_with_vizualization.py via la constante LLM_MODEL
- dans utils/dispatch_visualization.py via --llm-model
- ou via la variable d'environnement OPENAI_MODEL si 	ry_generate_llm_summary(...) est appelee directement sans argument explicite

## Comment le dashboard avec LLM fonctionne
Le pipeline suit les etapes suivantes :
1. Chargement des donnees prédites via load_predicted_inputs(...).
2. Chargement de l'etat du systeme via load_current_state(...).
3. Execution de l'optimiseur pour produire le plan horaire avec Pin, Pgo, PV, Pch, Pdis, Ebat, S.
4. Construction du dashboard Plotly via create_dispatch_dashboard(...).
5. Calcul des métriques synthetiques via build_visualization_frame(...) et summarize_dispatch(...).
6. Construction d'un payload de synthese via build_optimization_summary_payload(...).
7. Generation du texte naturel via 	ry_generate_llm_summary(...).
8. Insertion du texte dans le rapport HTML final avec save_dashboard_report_html(...).
Le LLM ne lit donc pas directement les graphes : Il recoit un resume structuré des resultats numériques et, selon le script utilisé, une partie du contexte d'optimisation.

## Ce que contient le payload envoye au LLM
Le payload construit dans optimization_summary.py contient notamment :
- les totaux energetiques sur 24h
- la demande servie
- les achats reseau Pin
- les ventes reseau Pgo
- la charge et la decharge batterie
- l'etat de charge initial, final et maximal
- la production PV et l'autoconsommation PV
- la flexibilite servie et coupee
- les couts d'achat, revenus de vente et cout net
- les pics horaires importants
- la repartition des regimes de fonctionnement
Quand le script complet de reporting est utilisé, le payload inclut aussi :
- une comparaison baseline vs optimiseur sur le cout et les emissions
- une description de la fonction objectif
- les variables de decision
- les contraintes principales
- les parametres du modele d'optimisation
C'est ce dernier point qui permet au texte generé d'expliquer non seulement ce qui s'est passé, mais aussi pourquoi l'optimiseur a pris certaines decisions lors de l'optim

## Strategie d'appel LLM et fallback
	ry_generate_llm_summary(...) applique la logique suivante :
1. tentative via client.responses.create(...)
2. si echec, nouvelle tentative via client.chat.completions.create(...)
3. si les deux appels echouent, ou si le package openai, la cle API, ou le nom de modele manque, retour d'un resume deterministe généré par build_rule_based_summary(...)
Le module supporte donc un fallback robuste
Attention toutefois : les scripts dashboard_24h_with_vizualization.py et utils/dispatch_visualization.py verifient la presence d'une cle avant d'executer la generation. En pratique, pour lancer ces scripts tels quels, il faut bien fournir une cle OpenRouter valide.

## Comment lancer les scripts
### Script de reporting complet
Depuis la racine du repo :
`ash
python energy_planner/src/reporting/dashboard_24h_with_vizualization.py
`
Parametres a ajuster directement dans le fichier :
- RUN_DATE
- LLM_MODEL
- LLM_BASE_URL
- OUTPUT_HTML
- OPEN_REPORT_IN_BROWSER
### Script utilitaire CLI
Depuis la racine du repo :
`ash
python utils/dispatch_visualization.py --llm-api-key your_key_here --llm-model openrouter/free --llm-base-url https://openrouter.ai/api/v1
`
Options disponibles :
- --run-date
- --llm-model
- --llm-api-key
- --llm-base-url
- --output-html
- --no-open
Note importante : dans l'état actuel du code, utils/dispatch_visualization.py expose --run-date mais force encore un date = date(2026, 3, 11) dans main(). Le rapport généré est donc actuellement lié à cette date tant que le script n'est pas modifie.

