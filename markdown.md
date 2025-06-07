# SGSimEval: A Comprehensive Multifaceted and Similarity-Enhanced Benchmark for Automatic Survey Generation Systems  

No Author Given  

No Institute Given  

Abstract. The growing interest in automatic survey generation (ASG), a task that traditionally required considerable time and effort, has been spurred by recent advances in large language models (LLMs). With advancements in retrieval-augmented generation (RAG) and the rising popularity of multi-agent systems (MASs), synthesizing academic surveys using LLMs has become a viable approach, thereby elevating the need for robust evaluation methods in this domain. However, existing evaluation methods suffer from several limitations, including biased metrics, a lack of human preference, and an over-reliance on LLMs-as-judges. To address these challenges, we propose SGSimEval, a comprehensive benchmark for Survey Generation with Similarity-Enhanced Evaluation that evaluates automatic survey generation systems by integrating assessments of the outline, content, and references, and also combines LLM-based scoring with quantitative metrics to provide a multifaceted evaluation framework. In SGSimEval, we also introduce human preference metrics that emphasize both inherent quality and similarity to humans. Extensive experiments reveal that current ASG systems demonstrate human-comparable superiority in outline generation, while showing significant room for improvement in content and reference generation, and our evaluation metrics maintain strong consistency with human assessments. Our code, data, and results will be released upon paper acceptance.  

Keywords: Large Language Models · Automatic Survey Generation Evaluation Benchmark · Semantic Similarity.  

# 1 Introduction  

Academic survey papers synthesize vast literature into coherent narratives, providing crucial scientific overviews. The exponential growth of academic literature has made manual survey creation increasingly time-consuming, driving urgent demands for automation. Recent advances in Large Language Models (LLMs), Retrieval Augmented Generation (RAG [6]), and Multi-Agent Systems (MASs [7]) have significantly advanced autonomous literature review [2,14,16,1] and scientific discovery [12,22,25], making automated survey generation (ASG) increasingly feasible.  

Recent ASG systems have demonstrated promising capabilities. AutoSurvey [20] established foundational methodologies with structured workflows, while  

SurveyX [10] enhanced performance through preparation and generation phases. Advanced systems like LLMxMapReduce-V2 [19] tackle extremely long inputs, SurveyForge [24] focuses on outline heuristics, and InteractiveSurvey [21] introduces personalized generation with user customization.  

Despite these advances, ASG evaluation remains problematic. Current frameworks suffer from inconsistent metrics across studies, making cross-system comparisons impossible. While LLM-as-a-Judge approaches [26,13] have gained prominence, existing evaluations inadequately integrate traditional metrics with LLMbased assessments and fail to capture survey quality across structure, references, and content dimensions comprehensively.  

To address these limitations, we propose SGSimEval, a comprehensive benchmark for ASG systems through similarity-enhanced assessment. SGSimEval integrates outline quality, content adequacy, and reference appropriateness evaluations while combining LLM-based scoring with quantitative metrics. Our framework introduces two complementary approaches: (1) human-reference alignment and (2) quality-balanced assessment weighing semantic similarity against intrinsic content quality. SGSimEval incorporates 80 highly-cited survey papers from diverse domains, enabling comprehensive assessments through systematic evaluation.  

Extensive experiments demonstrate SGSimEval’s effectiveness in revealing performance variations across ASG approaches. Results show Computer Sciencespecialized systems like AutoSurvey and SurveyForge consistently outperform general-domain systems across all dimensions, with most ASG systems surpassing human performance in outline generation and CS-specific systems exceeding humans in content quality, though human-generated references maintain substantially higher quality. Human consistency evaluations through pairwise comparisons confirm that our similarity-enhanced framework provides more nuanced and reliable assessments compared to traditional approaches, with evaluation results aligning well with human preferences across structural organization, content creation, and reference curation dimensions.  

This work makes three contributions:  

– We present SGSimEval, a comprehensive benchmark integrating outline quality, content adequacy, and reference appropriateness assessments for ASG evaluation. – We introduce a similarity-enhanced framework with complementary approaches considering both structural alignment and intrinsic content quality, providing better alignment with human preference for survey evaluation. – We provide extensive experimental validation demonstrating our benchmark’s effectiveness in identifying meaningful differences between ASG systems.  

# 2 Related Works  

# 2.1 Automated Survey Generation  

Large Language Models (LLMs), Retrieval Augmented Generation (RAG), and Multi-Agent Systems (MASs) have significantly advanced autonomous literature review [2,14,16,17,3] and scientific discovery [12,22,25,15,11], providing crucial support and inspiration for the implementation of automated survey generation (ASG) systems.  

Recent works in ASG systems have systematically addressed automated literature survey generation through various approaches. AutoSurvey [20] established foundational methodologies for comprehensive survey automation, introducing structured workflows from retrieval to evaluation that address LLMs’ context window limitations. Building upon this foundation, SurveyX [10] enhanced the framework by decomposing survey writing into distinct Preparation and Generation phases, incorporating AttributeTree pre-processing and achieving notable improvements in content quality (0.259) and citation quality (1.76) over existing systems.  

Addressing the challenge of processing extremely long inputs, LLMxMapReduceV2 [19] introduces convolutional test-time scaling strategies using stacked layers for progressive input understanding. Similarly, SurveyForge [24] focuses on outline heuristics and memory-driven generation, analyzing human-written outline structures and introducing SurveyBench for comprehensive evaluation across reference, outline, and content quality dimensions.  

Most recently, InteractiveSurvey [21] represents a paradigm shift toward personalized and interactive survey generation, enabling continuous user customization of reference categorization, outline, and content through an intuitive interface while maintaining superior content quality and time efficiency.  

# 2.2 Evaluation of ASG systems  

The evaluation of generative models has increasingly adopted LLM-as-a-Judge approaches [26,13,23,5], offering advantages in consistency and efficiency while significantly reducing human annotation costs. In the automated survey generation domain, this trend has been particularly valuable given the complexity and multifaceted nature of survey quality assessment.  

Recent advancements in ASG evaluation leverage LLMs alongside traditional metrics and human expert assessments across three core dimensions: structure, references, and content quality. AutoSurvey [20] employs a multi-LLM scoring strategy, achieving high consistency with human judgments in citation quality and content relevance, thereby reducing manual annotation costs. Similarly, SurveyForge [24] utilizes LLMs to rapidly assess outline rationality, improving structural evaluation efficiency by more than 30%. These LLM-based approaches are complemented by traditional metrics such as semantic similarity and IoU (reference overlap rate) [10], forming a multidimensional evaluation system that ensures model outputs align with both semantic logic and real-world academic scenarios.  

Expert evaluations remain crucial for providing granular critical analysis and structural coherence assessments. SurveyForge [24] employs a panel of 20 PhD student evaluators for comprehensive quality assessment, while LLMxMapReduceV2 [19] implements full-text citation verification mechanisms to ensure precise alignment between claims and original literature sources.  

Beyond quality assessment, efficiency benchmarks have become essential evaluation criteria. AutoSurvey [20] demonstrates significant speed advantages (73.59 papers per hour versus 0.07 papers per hour for humans), while InteractiveSurvey [21] evaluates system usability through the System Usability Scale (SUS), providing operational guidance for real-world implementation.  

# 3 SGSimEval  

Figure 1 illustrates the comprehensive pipeline of SGSimEval, which processes both human-authored and ASG-generated surveys through five key stages: (a) Data Collection, (b) Topic Mining, (c) Decomposition, (d) Embedding Generation, and (e) Evaluation. Figure 3 provides an overview of the evaluation aspects included in our framework. Our approach begins with curating a high-quality dataset of human-authored surveys, which undergoes systematic decomposition into outlines, content, and references. These components are transformed into contextual embeddings and stored in a vector database for similarity computation. The evaluation stage employs three assessment approaches: Vanilla evaluation using traditional metrics, Human-as-Perfect similarity weighting, treating human content as ideal references, and Balanced similarity weighting, considering both semantic alignment and actual human content quality. This enables a comprehensive assessment of ASG systems across structural, semantic, and quality dimensions while incorporating human preference alignment through automated similarity-based weighting mechanisms.  

# 3.1 Dataset Curation  

The SGSimEval dataset is curated through a multi-stage process. Initially, $I = 8 0$ highly-cited survey papers are collected and parsed to extract their topics, outlines, content, and references. Subsequently, contextual embeddings $O _ { i , m _ { o } }$ , $C _ { i , m _ { c } }$ , and $R _ { i , m _ { r } }$ for the extracted components are generated and stored in a vector database for further calculation. Each human-authored survey is represented as $S _ { i } = \{ T _ { i } , O _ { i } , C _ { i } , R _ { i } \}$ , where $T _ { i }$ denotes topic, $O _ { i }$ denotes outline, $C _ { i }$ denotes content, $R _ { i }$ denotes references, and $i$ ranges from 1 to $I$ .  

Data Collection. Our SGSimEval dataset comprises 80 highly-cited literature surveys from arXiv $^ { 1 }$ spanning various domains, all published within the last three years. The surveys are ranked by citation statistics from Semantic Scholar [9], and we select the top 80 for our dataset. The PDFs of these survey papers undergo initial processing via MinerU [18], an efficient PDF parsing tool. Subsequently, a rule-based extraction approach facilitates the retrieval of key information, including outlines, content, and references. To preserve the hierarchical structure of the outlines, we align and restore the original section levels for documents lacking this information using LLM. For the content, text segmentation follows chapter divisions. Given the heterogeneity in citation formats across domains, conferences, and journals, we implement specialized extraction rules for mainstream citation styles, achieving a 97.77% success rate in reference extraction across a test set of 371 survey articles.  

![](https://cdn-mineru.openxlab.org.cn/extract/242a8c3a-3074-410f-ae02-74a61fbf9a43/7df81e63e1fd160f88e9838f9f4b660c6753f368cd5b6af58be07d69e1e0f92c.jpg)  
Fig. 1. SGSimEval processes all the human-authored surveys and ASG-generated surveys through a) Data Collection, b) Topic Mining, c) Decomposition, d) Embedding Generation, and e) Evaluation. Three types of assessment results are marked as Vanilla, Balanced, and Human-as-Perfect.  

Topic Mining. For all 80 survey papers, we employ LLM to extract high-level topics from their titles. Each title is fed to the model to generate a concise topic label. These labels serve as the primary input for the evaluation of the ASG generation systems (e.g., "Large Language Models Meet NL2Code: A Survey" is labeled as "Natural Language to Code Generation with Large Language Models"). The model’s output is then manually verified for accuracy and relevance. This process ensures that the topics are not only relevant but also representative of the content within the survey papers.  

Decomposition. We decompose each survey into its constituent components: outline, content, and references. This decomposition process employs rule-based extraction approaches to systematically parse the structured information from each survey paper.  

Contextual Embedding Generation. As shown in Figure 2, to enable granular similarity comparison, we iteratively split each component type according to specific rules designed to preserve semantic coherence within individual units. For the outlines, a hierarchical section tree is constructed. Each path from the root to a leaf node is extracted. Subsequently, paths corresponding to sibling leaf nodes are aggregated to form distinct documents, for which embedding vectors are then generated. For the textual content, the content of each section is treated as an individual document, and corresponding vector representations are created. Similarly, each bibliographic reference is processed as a discrete document, and its respective embedding vector is generated. All the split embeddings of the outlines, content, and references $O _ { i , m _ { o } }$ , $\boldsymbol { C } _ { i , m _ { c } }$ , and $R _ { i , m _ { r } }$ are then stored in ChromaDB $^ 2$ for further retrieval and comparison.  

![](https://cdn-mineru.openxlab.org.cn/extract/242a8c3a-3074-410f-ae02-74a61fbf9a43/a7cd184e7add0d296bd853c52a50ce8011b94ceb8d38e5ae82c4935d07da1804.jpg)  
Fig. 2. The contextual embedding generation involves splitting the outline, content, and reference into outline paths, content sections, and reference entries, and encoding them into sentence embeddings.  

# 3.2 Evaluation Framework  

SGSimEval-Outline. SGSimEval-Outline evaluates the structural quality and logical coherence of generated survey outlines through hierarchical analysis and LLM-based assessment.  

1. Hierarchy: The hierarchical structure of the generated outlines is evaluated to ensure logical organization and coherence. This is achieved by a depthbased weighted scoring mechanism. For each non-leaf (parent) node $i$ in the outline, a local score $\boldsymbol { L } _ { i }$ is determined. This score $\boldsymbol { L } _ { i }$ represents the proportion of its child nodes that are considered coherent with the parent node, as assessed by LLM. Specifically, the LLM evaluates whether each child logically follows from its parent. Each node $i$ is assigned a weight $w _ { i }$ that is inversely proportional to its depth $d _ { i }$ , giving higher importance to nodes closer to the root. The weight is calculated as:  

$$
w _ { i } = \frac { D _ { m a x } - d _ { i } + 1 } { D _ { m a x } }
$$  

![](https://cdn-mineru.openxlab.org.cn/extract/242a8c3a-3074-410f-ae02-74a61fbf9a43/335a7fb0a33d2fad2a1f5eb31d4d51c1ec623fb0b147e4f14519ddbb3b128081.jpg)  
Fig. 3. The multifaceted evaluation aspects included in SGSimEval.  

where $D _ { m a x }$ is the maximum depth of the outline. The global hierarchy score $H _ { \mathrm { o u t l i n e } }$ is then computed as the weighted average of the local scores of all non-leaf nodes $P$ :  

$$
H _ { \mathrm { o u t l i n e } } = \left( \frac { \sum _ { i \in P } L _ { i } \cdot w _ { i } } { \sum _ { i \in P } w _ { i } } \right) \times 1 0 0 \
$$  

This score reflects the structural integrity of the generated outline.  

2. LLM-Assessed Quality: Following existing studies [24], we also employ the LLM to assess the overall quality of the survey paper outline by LLMgenerated criteria.  

SGSimEval-Content. SGSimEval-Content assesses the quality of generated survey content through citation faithfulness evaluation and comprehensive LLMbased quality assessment across multiple dimensions.  

1. Faithfulness: The faithfulness of the generated content evaluates how well cited references support the claims made in the text. For each sentence containing citations, we use an LLM to determine whether the cited references support the statement. The faithfulness score $F _ { \mathrm { c o n t e n t } }$ is calculated as:  

$$
F _ { \mathrm { c o n t e n t } } = { \frac { \mathrm { N u m b e r ~ o f ~ S u p p o r t i n g ~ R e f e r e n c e s } } { \mathrm { T o t a l ~ N u m b e r ~ o f ~ C i t e d ~ R e f e r e n c e s } } } \times 1 0 0 \
$$  

This measures how well the content is backed by appropriate citations.  

2. LLM-Assessed Quality: Using LLMs to evaluate survey content based on LLM-generated assessment criteria is commonly employed in existing works [20,24,19]. However, different studies focus on limited specific aspects. For example, [20,10,21] emphasize Coverage, Structure, and Relevance of survey content, while [19] concentrate more on Criticalness and Language. To achieve a more comprehensive assessment, we review prior works and consolidate all assessment metrics. Specifically, in SGSimEval, we evaluate the content quality of survey papers using an LLM across five dimensions as shown in Table 1.  

Table 1. LLM assessment dimensions for content quality evaluation, their sources and descriptions.   


<html><body><table><tr><td>Dimension Source</td><td></td><td>Description</td></tr><tr><td>Coverage</td><td>[20]</td><td>Measures how thoroughly the core topics are addressed.</td></tr><tr><td>Structure</td><td>[20]</td><td>Evaluates the logical organization of information.</td></tr><tr><td>Relevance</td><td>[20]</td><td>Considers the pertinence of the content to the survey topic.</td></tr><tr><td>Criticalness</td><td>[19]</td><td>Examines the degreeof critical analysisand insight provided.</td></tr><tr><td>Language</td><td>[19]</td><td>Assesses the clarity and professionalism of the writing.</td></tr></table></body></html>  

SGSimEval-Reference. SGSimEval-Reference evaluates the quality and relevance of generated bibliographies through supportiveness analysis and comprehensive LLM-based reference assessment.  

1. Supportiveness: The supportiveness of the references evaluates how well the bibliography supports the survey’s main topic. For each reference in the generated bibliography, we use an LLM to determine whether it is relevant to the survey topic. The supportiveness score $S _ { \mathrm { r e f e r e n c e } }$ is calculated as:  

$$
S _ { \mathrm { r e f e r e n c e } } = { \frac { \mathrm { N u m b e r ~ o f ~ R e l e v a n t ~ R e f e r e n c e s } } { \mathrm { T o t a l ~ N u m b e r ~ o f ~ R e f e r e n c e s } } } \times 1 0 0
$$  

This measures how well the bibliography aligns with the survey’s subject matter.  

2. LLM-Assessed Quality: To evaluate the overall quality of the references of a survey paper, we also employ LLM for assessment with a specific LLMgenerated criterion. The LLM assesses an overall score to the references based on their relevance, diversity, and adequacy in supporting the survey content.  

# 3.3 Similarity-Enhanced Evaluation Framework  

Human preferences serve as valuable complements to LLM scores [26,4]. To introduce human preference alignment while automating the evaluation process, we further apply analogous processing to ASG-generated surveys $\{ S _ { i } ^ { a } \} _ { i = 0 } ^ { I - 1 }$ $S _ { i } ^ { a } = \{ T _ { i } ^ { a } , O _ { i } ^ { a } , C _ { i } ^ { a } , R _ { i } ^ { a } \}$ eference value of human-authored surveys in evalua$\{ S _ { i } ^ { h } \} _ { i = 0 } ^ { I - 1 }$ tion, we propose two similarity-enhanced approaches that incorporate semantic alignment between system-generated and human-authored content. The first approach, Human-as-Perfect Similarity Weighting, treats human-authored content as the ideal reference standard. The second approach, Balanced Similarity Weighting, provides a more nuanced evaluation by considering both semantic similarity and the actual quality scores of human-authored content.  

Human-as-Perfect Similarity Weighting. This scoring strategy emphasizes the importance of human-like quality by treating human-authored content as a perfect reference. We assess semantic alignment between system-generated and human-authored components through embedding-based similarity, where the embeddings of outlines, contents, and references $O _ { i } ^ { a }$ , $C _ { i } ^ { a }$ , and $\mathbf { \mathcal { R } } _ { i } ^ { a }$ from the ASG-generated survey are generated through the same pipeline (Section 3.1) as the human-authored survey $O _ { i } ^ { h }$ , $C _ { i } ^ { h }$ , and $R _ { i } ^ { h }$ .  

For each embedding entry $O _ { i , j _ { o } } ^ { a }$ , $C _ { i , j _ { c } } ^ { a }$ , and $R _ { i , j _ { r } } ^ { a }$ , we obtain the most similar outline path, content section, or reference $O _ { i , j _ { o } ^ { * } } ^ { h }$ , $C _ { i , j _ { c } ^ { * } } ^ { h }$ , and $R _ { i , j _ { r } ^ { * } } ^ { h }$ from the humanauthored survey, and average the top $n _ { o , c , r }$ similarity scores, which serve as the weighting factors, for the scores of $\{ O _ { i } ^ { a } , C _ { i } ^ { a } , R _ { i } ^ { a } \}$ respectively.  

The weighting factors are calculated as follows:  

$$
\sigma ^ { k } = \frac { 1 } { N ^ { k } } \sum _ { j = 1 } ^ { N ^ { k } } \cos \left( V _ { i , j } ^ { a , k } , V _ { i , j ^ { * } } ^ { h , k } \right) , \quad k \in \left\{ \mathrm { o , c , r } \right\}
$$  

where $\mathbf { \mathbf { } } V ^ { k }$ represents $\{ O , C , R \}$ for outline, content, and references respectively, $N ^ { k }$ denotes the number of top similarity scores for each component type, and $V _ { i , j ^ { * } } ^ { h , k }$ are the embeddings of the most similar human-authored counterparts.  

The final evaluation score $Q _ { i } ^ { o , c , r ^ { \prime } }$ combines the similarity score ( $\sigma$ ) with the system output quality $\left( Q _ { i } ^ { o , c , r } \right)$ , assuming human-authored content as perfect with $Q _ { 0 } = 5$ :  

$$
Q _ { i } ^ { o , c , r ^ { \prime } } = \sigma _ { i } ^ { o , c , r } \cdot Q _ { 0 } + ( 1 - \sigma _ { i } ^ { o , c , r } ) \cdot Q _ { i } ^ { o , c , r }
$$  

Balanced Similarity Weighting. This approach provides a more balanced evaluation by considering both semantic similarity and actual quality scores of human-authored content. The similarity weighting factors $\sigma ^ { \mathrm { { o } } }$ , $\sigma ^ { \mathrm { c } }$ , and $\boldsymbol { \sigma } ^ { \mathrm { { r } } }$ are calculated using the same methodology as described above. However, rather than assuming human-authored content as perfect, this approach incorporates the actual quality scores of human-authored references.  

The final evaluation score $Q _ { i } ^ { o , c , r ^ { \prime } }$ for each component type combines the similarity score with both human reference quality and system output quality:  

$$
Q _ { i } ^ { o , c , r ^ { \prime \prime } } = \sigma _ { i } ^ { o , c , r } \cdot Q _ { \mathrm { h u m a n } , i } ^ { o , c , r } + ( 1 - \sigma _ { i } ^ { o , c , r } ) \cdot Q _ { i } ^ { o , c , r }
$$  

where Qoh,uc,mra n,i represents the actual quality score of the human-authored reference for the corresponding component type and metric. This formulation provides a more realistic assessment by considering the actual quality variations in human-authored content rather than treating it as uniformly perfect.  

# 4 Experiments  

# 4.1 Experiment Settings  

For all the LLM-as-Judge scoring and NLI tasks, we use Qwen-Plus-2025-04- $2 8 ^ { 3 }$ with a temperature of 0.5 and a context window of 128k. For all vector models, we use Alibaba’s Text-Embedding-V33 with 1024 dimensions. As detailed in Section 3.3, for the similarity scores, we consider $N ^ { o } = N ^ { c } = 5$ for outline, content and $N ^ { r } = 2 0$ for reference. These values were chosen to balance precision and recall; a smaller $N$ for outline and content focuses on the most critical structural and semantic similarities, while a larger $N$ for references allows for a more comprehensive comparison of the cited literature.  

To compare existing Automatic Survey Generation (ASG) systems, we evaluate outputs from five representative tools: InteractiveSurvey [21], SurveyX [10], AutoSurvey [20], SurveyForge [24], and LLMxMapReduce-V2 [19], using the same topic set as our benchmark. Due to open-source and deployment constraints, AutoSurvey and SurveyForge are restricted to the Computer Science domain. We achieved an overall successful collection rate of 85.38% across all systems.  

# 4.2 Evaluated ASG Systems  

We evaluate both human-written surveys and machine-generated surveys from five influential ASG systems:  

InteractiveSurvey [21] enables real-time modifications with user-uploaded papers, customizable categorizations, and iterative refinement of outlines and multi-modal content.  

SurveyX [10] employs a two-phase process: reference retrieval and AttributeTree construction, followed by outline creation and RAG-based content refinement.  

AutoSurvey [20] follows a four-step process: embedding-based outline generation, parallel LLM-based drafting, content refinement with citation checking, and MultiLLM-as-Judge selection.  

SurveyForge [24] generates surveys through heuristic outline generation, memorydriven Scholar Navigation Agent (SANA) for literature retrieval, and content synthesis with refinement.  

LLMxMapReduce [19] employs survey tree construction, entropy-driven skeleton refinement with Best-of-N self-refinement, and topology-aware content generation.  

# 5 Results  

# 5.1 Performance Comparison across Systems  

We present the performance difference among the evaluated ASG systems in Table 2, Figures 4 and 5.  

CS-specific systems (AutoSurvey and SurveyForge [20,24]) consistently outperform general-domain systems (InteractiveSurvey, SurveyX, and LLMxMapReduceV2 [21,10,19]) across outline, content, and reference dimensions. This superiority likely stems from domain-specific databases and the abundance of high-quality CS papers on platforms like arXiv.  

Within CS-specific systems, SurveyForge achieves better outline structure and reference quality than AutoSurvey, with comparable content quality. SurveyForge’s heuristic learning-based outline generation analyzes human-written survey structures and leverages domain literature to guide LLM output [24]. Its memory-driven scholar navigation agent retrieves high-quality, relevant papers, ensuring reference authority and relevance.  

Table 2. Performance of different systems using SGSimEval across outline, content, and reference metrics. Results are shown for vanilla, SGSimEval-B (Balanced Weighting), and SGSimEval-HP (Human as Perfect weighting) configurations. All scores are normalized to a 0-5 scale (quantitive scores with 0-100 scale are marked with $" \ast "$ ). Subscript values indicate the original percentage scores before normalization. For Outline and Reference, L1 represents overall quality. For Content, L1-L5 correspond to Coverage, Structure, Relevance, Language, and Criticalness, respectively.   


<html><body><table><tr><td rowspan="2">System</td><td rowspan="2">Metric</td><td rowspan="2">Avg.</td><td colspan="2">Outline</td><td colspan="6">Content</td><td colspan="2">Reference</td></tr><tr><td>*Structure</td><td></td><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td><td>*Faithfulness</td><td>L1</td><td>*Supportiveness</td></tr><tr><td colspan="10">Domain-Specific Systems (Computer Science)</td><td colspan="3"></td></tr><tr><td rowspan="3">AutoSurvey</td><td>vanilla</td><td>3.85</td><td>4.00</td><td>4.6893.57</td><td></td><td></td><td>5.00 4.60 5.00 5.00 4.90</td><td></td><td></td><td>3.4969.75</td><td>3.70</td><td>3.3967.74</td></tr><tr><td>SGSimEval-B</td><td>4.00</td><td>4.02</td><td>4.7795.31</td><td></td><td></td><td>5.00 4.27 5.00 4.61 4.83</td><td></td><td></td><td>4.0881.58</td><td>4.03</td><td>3.1462.89</td></tr><tr><td>SGSimEval-HP</td><td>4.54</td><td>4.65</td><td>4.8997.70</td><td></td><td></td><td>5.00 4.86 5.00 5.00 4.96</td><td></td><td></td><td>4.4689.17</td><td>4.41</td><td>4.2885.55</td></tr><tr><td rowspan="3">SurveyForge</td><td>vanilla</td><td>4.33</td><td>4.00</td><td>4.8897.50</td><td>5.00 4.00</td><td></td><td>5.00 5.00 4.60</td><td></td><td></td><td>4.0480.79</td><td>4.60</td><td>4.0781.34</td></tr><tr><td>SGSimEva-B</td><td>4.17</td><td>4.00</td><td>4.8396.67</td><td></td><td></td><td>5.00 4.07 5.00 4.60 4.73</td><td></td><td></td><td>4.2885.65</td><td>4.41</td><td>3.3967.75</td></tr><tr><td>SGSimEval-HP</td><td>4.75</td><td>4.59</td><td>4.9598.97</td><td>5.00 4.67 5.00 5.00 4.87</td><td></td><td></td><td></td><td></td><td>4.6793.37</td><td>4.84</td><td>4.62g2.37</td></tr><tr><td colspan="10">General Domain Systems</td><td colspan="3"></td></tr><tr><td rowspan="3">InteractiveSurvey</td><td>vanilla</td><td>3.31</td><td>3.60</td><td>4.0681.28</td><td>4.70 3.94 4.86 4.87 4.04</td><td></td><td></td><td></td><td></td><td>3.5971.88</td><td>3.23</td><td>2.2845.64</td></tr><tr><td>SGSimEvaL-B</td><td>3.49</td><td>3.69</td><td>4.3486.82</td><td></td><td></td><td>4.79 4.13 4.94 4.65 4.30</td><td></td><td></td><td>3.7975.80</td><td>3.45</td><td>2.3446.71</td></tr><tr><td>SGSimEval-HP</td><td>4.00</td><td>4.29</td><td>4.5791.31</td><td></td><td></td><td>4.87 4.56 4.94 4.95 4.59</td><td></td><td></td><td>4.4288.31</td><td>3.66</td><td>3.0160.29</td></tr><tr><td rowspan="3">LLMxMapReduce-V2</td><td>vanilla</td><td>3.28</td><td>4.02</td><td>4.8897.63</td><td></td><td></td><td></td><td></td><td></td><td>2.2344.59</td><td>2.32</td><td>2.7154.30</td></tr><tr><td>SGSimEval-B</td><td>3.56</td><td>3.94</td><td>4.7194.26</td><td></td><td></td><td>4.86 4.02 5.00 4.40 4.33 4.84 4.14 4.99 4.43 4.42</td><td></td><td></td><td>3.2565.00</td><td>3.15</td><td>2.7154.27</td></tr><tr><td>SGSimEval-HP</td><td>4.19</td><td>4.58</td><td>4.9598.98</td><td></td><td></td><td>4.94 4.64 5.00 4.77 4.75</td><td></td><td></td><td>4.0079.92</td><td>3.41</td><td>3.6472.77</td></tr><tr><td rowspan="3">SurveyX</td><td>vanilla</td><td>3.50</td><td>3.78</td><td>4.8496.74</td><td></td><td>4.06 3.35</td><td>5.00 3.97 3.61</td><td></td><td></td><td>3.7073.92</td><td>3.01</td><td>1.9539.04</td></tr><tr><td>SGSimEval-B</td><td>3.61</td><td>3.83</td><td>4.6993.87</td><td></td><td></td><td>4.51 3.84 4.99 4.24 4.09</td><td></td><td></td><td>3.7775.43</td><td>3.70</td><td>2.3647.18</td></tr><tr><td>SGSimEval-HP</td><td>4.31</td><td>4.49</td><td>4.9398.63</td><td></td><td>4.59 4.28</td><td>5.00</td><td>4.55 4.38</td><td></td><td>4.4589.04</td><td>4.05</td><td>3.5470.88</td></tr><tr><td colspan="10">Human Written</td><td colspan="3"></td></tr><tr><td>Human</td><td>vanilla</td><td>3.74|3.89</td><td></td><td>4.5691.10</td><td></td><td>4.84 4.21 4.99 4.45 4.49</td><td></td><td></td><td></td><td>3.8276.42</td><td>|4.30</td><td>2.8456.70</td></tr></table></body></html>  

Among general-domain systems, LLMxMapReduce-V2 excels in outline generation through its skeleton-driven multi-stage process with entropy-driven convolution mechanisms that enhance logical coherence [19]. However, its reliance on online search compromises reference quality. SurveyX demonstrates superior content quality via its two-stage process: preparation stage screening with the AttributeTree method for key information extraction, followed by RAG-based generation leveraging this curated knowledge base [10].  

# 5.2 Performance Comparison between Human and ASG Systems  

Then, we also identify significant differences between ASG systems and humangenerated surveys across evaluation dimensions, as shown in Table 2.  

Specifically, most ASG systems surpass humans in outline quality, as outline generation follows recognizable structural patterns that systems can effectively learn and replicate. For content quality, CS-specific systems [20,24] outperform humans, while general-domain systems [21,10,19] achieve comparable performance. This superiority of domain-specific systems stems from their tailored knowledge bases and specialized algorithms.  

![](https://cdn-mineru.openxlab.org.cn/extract/242a8c3a-3074-410f-ae02-74a61fbf9a43/09d6e0a8e810bba3177baace6d129a2f9c296ac3d4085a6c7d7eaa4fb90c6a1b.jpg)  
Fig. 4. Performance comparison for Computer Science domain systems on vanilla, SGSimEval-B, and SGSimEval-HP configurations  

Table 3. Results of the similarity comparison between ASG Systems and human-written surveys   


<html><body><table><tr><td>System</td><td>Outline Content Reference Avg.</td><td></td><td></td></tr><tr><td>AutoSurvey</td><td>0.63</td><td>0.65 0.56</td><td>0.61</td></tr><tr><td>SurveyForge</td><td>0.59</td><td>0.67 0.61</td><td>0.62</td></tr><tr><td>InteractiveSurvey</td><td>0.54</td><td>0.58 0.32</td><td>0.48</td></tr><tr><td>LLMxMapReduce-V2</td><td>0.58</td><td>0.64 0.42</td><td>0.54</td></tr><tr><td>SurveyX</td><td>0.60</td><td>0.57 0.56</td><td>0.58</td></tr><tr><td>Human</td><td>1.00</td><td>1.00 1.00</td><td>1.00</td></tr></table></body></html>  

However, human-generated references maintain substantially higher quality than current ASG systems. Humans excel through sophisticated information retrieval, critical source evaluation, and meticulous citation accuracy, whereas ASG systems have yet to fully develop in these areas.  

Table 3 presents similarity scores between ASG outputs and human surveys. Outline similarity ranges from 0.54-0.63, with AutoSurvey achieving the highest score (0.63). Content similarity shows higher overall scores (0.57-0.67), with CS-specific systems leading: SurveyForge (0.67) and AutoSurvey (0.65) versus general-domain systems (0.57-0.64). Reference similarity exhibits the greatest variation (0.32-0.61), with SurveyForge leading (0.61) and InteractiveSurvey lagging (0.32). These patterns confirm that reference selection remains the most challenging aspect for current ASG systems.  

# 5.3 Human Consistency  

To assess consistency between human preference and our evaluation framework, we conduct LLM-based pairwise comparisons following [8]. Using Qwen2.5-72BInstruct $^ 3$ , Qwen-Plus-2025-04-283, and Qwen-Turbo-2025-04-28 $^ 3$ , we compute average win rates for each ASG system against human baselines across outline, content, and reference dimensions, as shown in Table 4.  

![](https://cdn-mineru.openxlab.org.cn/extract/242a8c3a-3074-410f-ae02-74a61fbf9a43/6d5b88e124dc85855137e568496325f2ff7c1151fb5fe6d1abcd39ce40d9ed3b.jpg)  
Fig. 5. Performance comparison for general domain systems on vanilla, SGSimEval-B, and SGSimEval-HP configurations  

Table 4. ASG systems’ Win Rate against human baselines.   


<html><body><table><tr><td>System</td><td>Outline</td><td></td><td>Content Reference</td></tr><tr><td>AutoSurvey</td><td>[0.73±0.15</td><td>0.30±0.17</td><td>0.10±0.10</td></tr><tr><td>SurveyForge</td><td></td><td>[0.63±0.15|0.33±0.35|</td><td>0.33±0.32</td></tr><tr><td>InteractiveSurvey</td><td></td><td>[0.44±0.11|0.37±0.15</td><td>0.07±0.06</td></tr><tr><td>LLMxMapReduce-V2|0.73±0.05|0.45±0.09</td><td></td><td></td><td>0.02±0.00</td></tr><tr><td>SurveyX</td><td>[0.63±0.12 |0.30±0.21</td><td></td><td>0.14±0.06</td></tr></table></body></html>  

The pairwise comparison results align with our quantitative metrics. ASG systems achieve higher win rates in outline generation, confirming their effectiveness in structural organization. However, win rates for content quality and reference selection indicate that ASG systems remain competitive but require improvement to consistently match human performance. This highlights the ongoing challenge of achieving human-level proficiency in nuanced content creation and reference curation, where domain expertise and critical judgment remain essential.  

# 6 Conclusion and Future Work  

In this paper, we propose SGSimEval, a comprehensive benchmark for evaluating automated survey generation systems with three key contributions: (1) a multidimensional evaluation framework that systematically assesses structural organization, content adequacy, and reference appropriateness; (2) a similarityenhanced evaluation approach that balances human-authored references with semantic alignment; and (3) extensive experimental validation across five representative ASG systems. Our results demonstrate that CS-specialized systems consistently outperform general-domain approaches, with most ASG systems exceeding human performance in outline generation, though significant challenges remain in reference curation.  

Future work includes expanding domain coverage beyond computer science, improving semantic evaluation metrics through hierarchical and multimodal embeddings, and developing dynamic assessment frameworks that adapt to evolving academic standards. We also aim to implement real-time evaluation capabilities and optimize computational efficiency to enhance scalability for large-scale survey generation systems.  

# References  

1. Agarwal, S., Laradji, I.H., Charlin, L., Pal, C.: LitLLM: A Toolkit for Scientific Literature Review (Feb 2024). https://doi.org/10.48550/arXiv.2402.01788, http: //arxiv.org/abs/2402.01788, arXiv:2402.01788 [cs]   
2. Agarwal, S., Sahu, G., Puri, A., Laradji, I.H., Dvijotham, K.D., Stanley, J., Charlin, L., Pal, C.: Litllm: A toolkit for scientific literature review (2025), https://arxiv. org/abs/2402.01788   
3. Ali, N.F., Mohtasim, M.M., Mosharrof, S., Krishna, T.G.: Automated literature review using nlp techniques and llm-based retrieval-augmented generation (2024), https://arxiv.org/abs/2411.18583   
4. Bai, G., Liu, J., Bu, X., He, Y., Liu, J., Zhou, Z., Lin, Z., Su, W., Ge, T., Zheng, B., Ouyang, W.: Mt-bench-101: A fine-grained benchmark for evaluating large language models in multi-turn dialogues. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). p. 74217454. Association for Computational Linguistics (2024). https://doi.org/10.18653/v1/ 2024.acl-long.401, http://dx.doi.org/10.18653/v1/2024.acl-long.401   
5. Chen, H., Xiong, M., Lu, Y., Han, W., Deng, A., He, Y., Wu, J., Li, Y., Liu, Y., Hooi, B.: Mlr-bench: Evaluating ai agents on open-ended machine learning research (2025), https://arxiv.org/abs/2505.19955   
6. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, H., Wang, H.: Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997 2 (2023)   
7. Han, S., Zhang, Q., Yao, Y., Jin, W., Xu, Z., He, C.: Llm multi-agent systems: Challenges and open problems. arXiv preprint arXiv:2402.03578 (2024)   
8. Idahl, M., Ahmadi, Z.: Openreviewer: A specialized large language model for generating critical scientific paper reviews (2025), https://arxiv.org/abs/2412.11948   
9. Kinney, R., Anastasiades, C., Authur, R., Beltagy, I., Bragg, J., Buraczynski, A., Cachola, I., Candra, S., Chandrasekhar, Y., Cohan, A., Crawford, M., Downey, D., Dunkelberger, J., Etzioni, O., Evans, R., Feldman, S., Gorney, J., Graham, D., Hu, F., Huff, R., King, D., Kohlmeier, S., Kuehl, B., Langan, M., Lin, D., Liu, H., Lo, K., Lochner, J., MacMillan, K., Murray, T., Newell, C., Rao, S., Rohatgi, S., Sayre, P., Shen, Z., Singh, A., Soldaini, L., Subramanian, S., Tanaka, A., Wade, A.D., Wagner, L., Wang, L.L., Wilhelm, C., Wu, C., Yang, J., Zamarron, A., Zuylen, M.V., Weld, D.S.: The semantic scholar open data platform (2025), https://arxiv.org/abs/2301.10140   
10. Liang, X., Yang, J., Wang, Y., Tang, C., Zheng, Z., Song, S., Lin, Z., Yang, Y., Niu, S., Wang, H., Tang, B., Xiong, F., Mao, K., li, Z.: Surveyx: Academic survey automation via large language models (2025), https://arxiv.org/abs/2502.14776   
11. Liu, C., Wang, C., Cao, J., Ge, J., Wang, K., Zhang, L., Cheng, M.M., Zhao, P., Li, T., Jia, X., Li, X., Li, X., Liu, Y., Feng, Y., Huang, Y., Xu, Y., Sun, Y., Zhou, Z., Xu, Z.: A vision for auto research with llm agents (2025), https: //arxiv.org/abs/2504.18765   
12. Lu, C., Lu, C., Lange, R.T., Foerster, J., Clune, J., Ha, D.: The ai scientist: Towards fully automated open-ended scientific discovery (2024), https://arxiv.org/abs/2408. 06292   
13. Que, H., Duan, F., He, L., Mou, Y., Zhou, W., Liu, J., Rong, W., Wang, Z.M., Yang, J., Zhang, G., Peng, J., Zhang, Z., Zhang, S., Chen, K.: Hellobench: Evaluating long text generation capabilities of large language models (2024), https://arxiv. org/abs/2409.16191   
14. Sami, A.M., Rasheed, Z., Kemell, K.K., Waseem, M., Kilamo, T., Saari, M., Duc, A.N., Systä, K., Abrahamsson, P.: System for systematic literature review using multiple ai agents: Concept and an empirical evaluation (2024), https://arxiv.org/ abs/2403.08399   
15. Schmidgall, S., Su, Y., Wang, Z., Sun, X., Wu, J., Yu, X., Liu, J., Liu, Z., Barsoum, E.: Agent laboratory: Using llm agents as research assistants (2025), https://arxiv. org/abs/2501.04227   
16. Susnjak, T., Hwang, P., Reyes, N., Barczak, A.L.C., McIntosh, T., Ranathunga, S.: Automating research synthesis with domain-specific large language model finetuning. ACM Transactions on Knowledge Discovery from Data 19(3), 139 (Mar 2025). https://doi.org/10.1145/3715964, http://dx.doi.org/10.1145/3715964   
17. Torres, J.P.F., Mulligan, C., Jorge, J., Moreira, C.: Promptheus: A human-centered pipeline to streamline slrs with llms (2024), https://arxiv.org/abs/2410.15978   
18. Wang, B., Xu, C., Zhao, X., Ouyang, L., Wu, F., Zhao, Z., Xu, R., Liu, K., Qu, Y., Shang, F., Zhang, B., Wei, L., Sui, Z., Li, W., Shi, B., Qiao, Y., Lin, D., He, C.: Mineru: An open-source solution for precise document content extraction (2024), https://arxiv.org/abs/2409.18839   
19. Wang, H., Fu, Y., Zhang, Z., Wang, S., Ren, Z., Wang, X., Li, Z., He, C., An, B., Liu, Z., Sun, M.: Llm $\times$ mapreduce-v2: Entropy-driven convolutional test-time scaling for generating long-form articles from extremely long resources (2025), https://arxiv.org/abs/2504.05732   
20. Wang, Y., Guo, Q., Yao, W., Zhang, H., Zhang, X., Wu, Z., Zhang, M., Dai, X., Zhang, M., Wen, Q., Ye, W., Zhang, S., Zhang, Y.: Autosurvey: Large language models can automatically write surveys (2024), https://arxiv.org/abs/2406.10252   
21. Wen, Z., Cao, J., Wang, Z., Guo, B., Yang, R., Liu, S.: Interactivesurvey: An llm-based personalized and interactive survey paper generation system (2025), https://arxiv.org/abs/2504.08762   
22. Weng, Y., Zhu, M., Bao, G., Zhang, H., Wang, J., Zhang, Y., Yang, L.: Cycleresearcher: Improving automated research via automated review (2025), https: //arxiv.org/abs/2411.00816   
23. Wu, Y., Mei, J., Yan, M., Li, C., Lai, S., Ren, Y., Wang, Z., Zhang, J., Wu, M., Jin, Q., Huang, F.: Writingbench: A comprehensive benchmark for generative writing (2025), https://arxiv.org/abs/2503.05244   
24. Yan, X., Feng, S., Yuan, J., Xia, R., Wang, B., Zhang, B., Bai, L.: Surveyforge: On the outline heuristics, memory-driven generation, and multi-dimensional evaluation for automated survey writing (2025), https://arxiv.org/abs/2503.04629   
25. Yuan, J., Yan, X., Feng, S., Zhang, B., Chen, T., Shi, B., Ouyang, W., Qiao, Y., Bai, L., Zhou, B.: Dolphin: Moving towards closed-loop auto-research through thinking, practice, and feedback (2025), https://arxiv.org/abs/2501.03916   
26. Zheng, L., Chiang, W.L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E.P., Zhang, H., Gonzalez, J.E., Stoica, I.: Judging llm-as-a-judge with mt-bench and chatbot arena (2023), https://arxiv.org/abs/2306.05685  