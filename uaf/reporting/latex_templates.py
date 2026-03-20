"""Jinja2 шаблоны LaTeX для всех секций отчёта UAF."""

# Основной шаблон документа
REPORT_MAIN = r"""
\documentclass[12pt,a4paper]{article}
\usepackage{fontspec}
\usepackage[main=russian,english]{babel}
\setmainfont{DejaVu Serif}
\setsansfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{fancyhdr}
\usepackage{graphicx}
\usepackage{listings}
\usepackage{float}
\usepackage{amsmath}

\geometry{
    a4paper,
    top=2.5cm,
    bottom=2.5cm,
    left=3cm,
    right=2.5cm
}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!60!black,
    urlcolor=blue!60!black,
    citecolor=green!60!black,
    pdfauthor={UAF AutoResearch Framework},
    pdftitle={Research Report: << task_title | latex_escape >>},
}

\lstset{
    basicstyle=\small\ttfamily,
    breaklines=true,
    frame=single,
    backgroundcolor=\color{gray!10},
    numbers=left,
    numberstyle=\tiny\color{gray},
    xleftmargin=2em,
    xrightmargin=1em,
}

\definecolor{passcolor}{rgb}{0.1,0.6,0.1}
\definecolor{failcolor}{rgb}{0.7,0.1,0.1}
\definecolor{warncolor}{rgb}{0.8,0.5,0.0}
\definecolor{partialcolor}{rgb}{0.5,0.5,0.8}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small UAF Research Report}
\fancyhead[R]{\small << session_id[:16] | latex_escape >>}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

\title{
    \Large\textbf{Research Report}\\[0.5em]
    \large << task_title | latex_escape >>\\[0.3em]
    \normalsize Session: \texttt{<< session_id | latex_escape >>}
}
\author{Universal AutoResearch Framework v2.0}
\date{<< created_at >>}

\begin{document}

\maketitle
\tableofcontents
\newpage

<< executive_summary_section >>

<< task_description_section >>

<< experiment_results_section >>

<< analysis_section >>

<< code_quality_section >>

<< reproducibility_section >>

\end{document}
"""

# Секция Executive Summary
EXECUTIVE_SUMMARY = r"""
\section{Executive Summary}

<% if executive_summary_text %>
<< executive_summary_text | md_to_latex >>
<% else %>
\textit{Секция Executive Summary не была сгенерирована Claude Code.}
<% endif %>
"""

# Секция Task Description
TASK_DESCRIPTION = r"""
\section{Task Description}

\begin{description}
    \item[Задача:] << task_title | latex_escape >>
    \item[Тип:] << task_type | latex_escape >>
    \item[Целевая метрика:] \texttt{<< target_metric | latex_escape >>}
        (<< metric_direction >>)
    \item[Датасет:] << dataset_path | latex_escape >>
    \item[Целевая переменная:] \texttt{<< target_column | latex_escape >>}
\end{description}

<% if problem_statement %>
\subsection*{Описание задачи}
<< problem_statement | latex_escape >>
<% endif %>
"""

# Секция Experiment Results
EXPERIMENT_RESULTS = r"""
\section{Experiment Results}

\subsection{Overview}

\begin{table}[H]
\centering
\caption{Сводка сессии}
\begin{tabular}{lrrrr}
\toprule
Итого runs & Completed & Failed & Partial & Best << target_metric >> \\
\midrule
<< total_runs >> & \textcolor{passcolor}{<< completed_runs >>}
    & \textcolor{failcolor}{<< failed_runs >>}
    & \textcolor{partialcolor}{<< partial_runs >>}
    & <% if best_value is not none %><< "%.6f" | format(best_value) >><% else %>---<% endif %> \\
\bottomrule
\end{tabular}
\end{table}

<% if ranked_runs %>
\subsection{Ranked Results}

\begin{longtable}{llrr}
\toprule
\# & Run Name & << target_metric >> & Status \\
\midrule
\endhead
<% for i, run in enumerate(ranked_runs) %>
<< i + 1 >> & \texttt{<< run.run_name | latex_escape >>}
    & <% if target_metric in run.metrics %><< "%.6f" | format(run.metrics[target_metric]) >>
      <% else %>---<% endif %>
    & \textcolor{passcolor}{\small success} \\
<% endfor %>
\bottomrule
\end{longtable}
<% endif %>

<% if failed_runs_list %>
\subsection{Failed Experiments}

\textit{Неудачные эксперименты включены согласно требованию antigoal~3
(не скрывать неудачи).}

\begin{longtable}{llp{7cm}}
\toprule
Run ID & Category & Причина \\
\midrule
\endhead
<% for run in failed_runs_list %>
\texttt{<< run.run_id[:12] >>}
    & << run.failure_category or "---" >>
    & \small << (run.failure_reason or "---") | latex_escape >> \\
<% endfor %>
\bottomrule
\end{longtable}
<% endif %>

<% if metric_figure_path %>
\subsection{Metric Progression}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{<< metric_figure_path >>}
\caption{Прогресс метрики << target_metric >> по итерациям}
\end{figure}
<% endif %>

<% if budget_figure_path %>
\subsection{Budget Burndown}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{<< budget_figure_path >>}
\caption{Budget Burndown: использование бюджета по итерациям}
\end{figure}
<% endif %>
"""

# Секция Analysis and Findings
ANALYSIS_SECTION = r"""
\section{Analysis and Findings}

<% if analysis_text %>
<< analysis_text | md_to_latex >>
<% else %>
\textit{Секция Analysis and Findings не была сгенерирована Claude Code.}
<% endif %>

<% if hypotheses %>
\subsection{Improvement Hypotheses}

\begin{description}
<% for h in hypotheses %>
    \item[\textbf{<< h.code >>} (P<< h.priority >>):] << h.description | latex_escape >>
    \textit{Основание: << h.evidence | latex_escape >>}
<% endfor %>
\end{description}
<% endif %>

<% if param_correlations %>
\subsection{Parameter Correlations}

\begin{table}[H]
\centering
\caption{Spearman корреляции параметр→<< target_metric >>}
\begin{tabular}{lrrr}
\toprule
Параметр & Spearman r & p-value & n \\
\midrule
<% for c in param_correlations[:10] %>
\texttt{<< c.param_name | latex_escape >>}
    & << "%.3f" | format(c.spearman_r) >>
    & << "%.4f" | format(c.p_value) if c.p_value else "---" >>
    & << c.n_samples >> \\
<% endfor %>
\bottomrule
\end{tabular}
\end{table}
<% endif %>
"""

# Секция Code Quality
CODE_QUALITY = r"""
\section{Code Quality Report}

<% if ruff_report %>
\begin{table}[H]
\centering
\caption{Отчёт ruff (<< ruff_report.ruff_version >>)}
\begin{tabular}{lr}
\toprule
Показатель & Значение \\
\midrule
Файлов проверено & << ruff_report.total_files >> \\
Чистых файлов & << ruff_report.clean_files >> \\
Clean rate & <% if ruff_report.target_met %>\textcolor{passcolor}{<% endif %><< "%.1f" | format(ruff_report.clean_rate * 100) >>\%<% if ruff_report.target_met %>}<% endif %> \\
Нарушений (после fix) & <% if ruff_report.total_violations == 0 %>\textcolor{passcolor}{0}<% else %>\textcolor{warncolor}{<< ruff_report.total_violations >>}<% endif %> \\
Файлов с нарушениями & << ruff_report.files_with_unfixable >> \\
Целевой показатель достигнут & <% if ruff_report.target_met %>\textcolor{passcolor}{Да}<% else %>\textcolor{failcolor}{Нет}<% endif %> \\
\bottomrule
\end{tabular}
\end{table}

<% if ruff_report.files_with_unfixable > 0 %>
\subsection*{Файлы с нарушениями}

\begin{longtable}{lr}
\toprule
Файл & Нарушений \\
\midrule
\endhead
<% for fr in ruff_report.files %><% if fr.violations_after_fix > 0 %>
\texttt{<< fr.file.name | latex_escape >>} & << fr.violations_after_fix >> \\
<% endif %><% endfor %>
\bottomrule
\end{longtable}
<% endif %>

<% else %>
\textit{RuffEnforcer не запускался или ruff\_report.json не найден.}
<% endif %>
"""

# Секция Reproducibility
REPRODUCIBILITY = r"""
\section{Reproducibility}

\begin{description}
    \item[Session ID:] \texttt{<< session_id | latex_escape >>}
    \item[Дата сессии:] << created_at >>
    \item[MLflow experiment:] \texttt{<< mlflow_experiment | latex_escape >>}
    \item[Claude модель:] \texttt{<< claude_model | latex_escape >>}
    \item[Random seed:] << random_seed or "---" >>
    \item[DVC коммит:] \texttt{<< (dvc_commit or "---") | latex_escape >>}
<% if git_sha %>
    \item[Git SHA:] \texttt{<< git_sha | latex_escape >>}
<% endif %>
\end{description}

<% if all_run_ids %>
\subsection*{MLflow Run IDs}

\begin{lstlisting}
<% for run_id in all_run_ids %><< run_id >>
<% endfor %>
\end{lstlisting}
<% endif %>

<% if requirements_lock %>
\subsection*{Зависимости (requirements.lock)}

\begin{lstlisting}
<< requirements_lock[:2000] >><% if requirements_lock | length > 2000 %>
... (truncated)<% endif %>
\end{lstlisting}
<% endif %>
"""
