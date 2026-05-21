Received 20 November 2024, accepted 22 January 2025, date of publication 29 January 2025, date of current version 18 February 2025. Digital Object Identifier 10.1109/ACCESS.2025.3536095

![image](p0_r2_image_0.jpg)

# A Multifaceted Vision of the Human-AI Collaboration: A Comprehensive Review

MAITE PUERTA-BELDARRAIN $ ^{1} $ , OIHANE GÓMEZ-CARMONA $ ^{1} $ , RUBÉN SÁNCHEZ-CORCUERA $ ^{2} $ , DIEGO CASADO-MANSILLA $ ^{2} $ , DIEGO LÓPEZ-DE-IPIÑA $ ^{2} $ AND LIMING CHEN $ ^{3} $ , (Senior Member, IEEE)

$ ^{1} $Deustotech, University of Deusto, 48007 Bilbao, Spain $ ^{2} $Faculty of Engineering, University of Deusto, 48007 Bilbao, Spain $ ^{3} $School of Computer Science and Technology, Dalian University of Technology, Dalian 116024, China

Corresponding author: Maite Puerta-Beldarrain (mpuerta004@deusto.es)

This work has been supported by grant IT1582-22 from the Basque Government, which recognizes Human-centric Computing for Smart Sustainable Communities and Environments (DEUSTEK5) as an excellent research group under the Basque university system. Besides, we acknowledge the Ministerio de Ciencia e Innovacion for Internet of People (IoP), under Grant No.: PID2020-119682RB-I00. Finally, this work has been partially supported by the European Commission through the Wearables and drones for City Socio-Environmental Observations and Behavioral Change (SOCIO-BEE) project Under Grant No. 101037648.

> ABSTRACT Human-AI collaboration has evolved into a complex, multidimensional paradigm shaped by research in various domains. Key areas such as human-in-the-loop systems, Interactive Machine Learning (IML), Hybrid Intelligence, and Human-Agent Interaction have significantly contributed to this development. However, these fields often lack cohesion, underscoring the need for a cohesive perspective to advance. This work addresses this gap by integrating insights from diverse aspects of collaboration to present a holistic approach to fostering effective and adaptive interactions between humans and artificial agents. It emphasizes empowering end-users with greater control and involvement in decision-making processes, thereby enhancing both the levels of interactivity and adaptability within intelligent systems. Moving beyond a focus on AI training techniques, this paper presents a broader perspective on incorporating human input into AI decision-making and learning processes, highlighting the importance of flexibility in systems and user engagement. The manuscript proposes a framework encompassing five levels of human integration and examines their relationship with core collaboration aspects, including the system purpose, participant expertise, and system proactivity. By synthesizing current knowledge on human-AI collaboration and outlining essential design principles, this work aims to advance the field and foster interdisciplinary collaboration among researchers, practitioners, and designers.

- INDEX TERMS Hybrid intelligence, human-AI collaboration, human-robot collaboration, human-machine collaboration, human-in-the-loop, interactive machine learning, human-machine symbiosis, human-centred AI.

## I. INTRODUCTION

Understanding this technological evolution requires a thorough examination of how technological systems have adapted to facilitate human involvement in decision-making processes. This transformation likely began with the development of the Ubiquitous Computing paradigm, which ignited the desire for greater technological adaptability and connectivity [4]. As these technologies became more widespread, the concept of Smart Environments emerged, designed to enhance our surroundings without demanding active human engagement [5], [6]. Building on this foundation, Ambient Intelligence (AmI) and Intelligent Environments (IE) further

Technology has rapidly evolved to become an integral component of contemporary life, progressively incorporating humans into its decision-making processes and actively integrating them within the technological framework [1], [2], [3]. The rising emphasis on human-centered technology has been a significant driver behind the advancements in AI's learning and reasoning capabilities.

The associate editor coordinating the review of this manuscript and approving it for publication was Zijian Zhang $ ^{1D} $ .

advanced the principle of user-centric technology, prioritizing interactive and responsive environments [7], [8]. In more recent developments, the concept of the Internet of People (IoP) has shifted its focus towards recognizing users as active contributors within technological ecosystems, rather than passive recipients of data-driven services [2], [9]. These evolving fields collectively illustrate that technology has been continuously advancing to meet users' increasing demand for a more immersive and integrated relationship with technology, positioning them in a central and participatory role.

the five clusters obtained based on the main characteristics of collaboration. Section VI describes the essential design principles users need for a long-lasting collaboration. Sections VII discuss the necessity to integrate human beings' opinions and knowledge in the design phase of future intelligent systems. Finally, in Section VIII the conclusions of this work are presented.

## II. A MULTIFACETED OVERVIEW OF THE COLLABORATION

Generally, collaboration is understood as two or more entities working together, sharing responsibility for solving problems, and actively interacting to achieve a common goal [16]. The primary purpose of collaboration is to leverage the strengths of each entity to compensate for the limitations of others, enabling achievements beyond what each could accomplish independently [12], [14], [17], [18]. However, this definition can vary significantly depending on the specific research domain. This section explores the benefits and challenges of collaboration and examines the different perspectives that this concept encompasses across various fields.

The progression of technology has significantly influenced the concept of collaboration across multiple disciplines over the years [10], [11]. In this context, collaboration is defined as joint interactions between technology and its users aimed at achieving a specific outcome. This notion of collaboration has evolved, taking on various terminologies and attributes depending on the particular field of study. This variation has led to a fragmentation of the concept across different domains, with each area focusing on unique dimensions of collaborative processes [12], [13], [14], [15]. Such fragmentation has contributed to a lack of coherence in the literature, creating obstacles to developing a unified understanding of the different types of collaboration, their requirements, and the nuances in integrating users into the decision-making process.

Artificial Intelligence (AI) has demonstrated remarkable computational capabilities, excelling in tasks that involve recognizing patterns, handling unforeseen scenarios, and adapting to diverse conditions using advanced techniques like zero-shot learning and pre-training models [19], [20], [21]. These capabilities enable AI to identify complex patterns in large datasets and process information at speeds that far exceed human abilities, all while remaining unaffected by factors such as stress, motivation, or fatigue [22].

This paper aims to bridge these diverse viewpoints by presenting a unified framework that facilitates the development of human-AI collaborative technologies based on a holistic understanding of the types and requirements of collaboration. It strives to synthesize insights from various research areas to establish a balanced, productive, and sustainable model of collaboration between humans and artificial intelligence (AI). The core contributions of this paper include:

However, despite its strengths, AI's performance is heavily dependent on the quality of its training data. Inaccuracies or biases in this data can lead to flawed outcomes, particularly in sensitive areas such as race, gender, ethics, and politics [20], [23]. Furthermore, AI often lacks a deep understanding of language and context, which can result in misinterpretations that undermine the accuracy of decision-making processes [20]. In contrast, humans inherently possess contextual awareness and ethical reasoning, allowing them to navigate ambiguities and make more nuanced decisions than the machines [24].

- A comprehensive overview of AI techniques and subfields that are instrumental in creating interactive technologies.

- A refined classification system categorizing five groups of human-AI collaboration, providing a structured guide to understanding the different types of collaboration and their distinctive features. This system is organized around key aspects of collaboration, such as the degree of human involvement, level of interaction, and collaboration type.

While the potential for collaboration between humans and AI is substantial due to their complementary strengths, several significant challenges must be addressed to ensure its success. For effective collaboration, both AI systems and human participants need to understand each other's roles, integrate each other's capabilities into their strategies, and communicate clearly to avoid misunderstandings [25]. Technical barriers also emerge from the fundamentally different ways humans and AI process information, which can create obstacles to seamless communication [25], [26]. Additionally, the complexity and opacity of many AI models—often referred to as black-box systems—pose challenges in terms of transparency, trust, and acceptance within collaborative environments [27], [28]. Nevertheless,

- Design principles for developing future human-centered collaborative technologies. These principles aim to expand the narrow focus of existing research by fostering a broader approach that enhances the comprehension of human needs within collaborative systems.

The remainder of this paper is organized as follows. Section II highlights several research areas related to humanAI collaboration to illustrate the necessity of developing a shared perspective on this concept. Section III describes the Methodology employed for the literature review. Section IV explains the techniques to develop an interactive approach between users and intelligent systems. Section V presents

interaction. To develop a comprehensive understanding of the various forms of collaboration, this paper will analyze the degrees of human involvement within these collaborative approaches and identify the key design principles necessary for enabling effective human participation.

fostering collaboration between humans and AI remains one of the most promising pathways for enhancing the capabilities of both entities.

This approach, despite its inherent challenges, offers considerable advantages by leveraging the unique capabilities of both humans and AI technologies. A major issue, however, lies in achieving a unified understanding of the diverse research fields that apply the concept of collaboration, especially in pinpointing the specific attributes that characterize collaboration within each field:

![image](p2_r5_image_1.jpg)

- Human-in-the-loop (HiTL): Focuses on a human-centered approach to collaboration by directly involving individuals in AI systems' decision-making processes [29], [30], [31].

- Interactive Machine Learning (IML): Highlights the importance of creating interactive communication channels between users and AI models, emphasizing user engagement throughout various stages of AI development [32].

- Human-Robot Collaboration: Centers on developing robots designed to work collaboratively with humans in physical spaces to achieve common objectives [33].

- Collaborative Interactive Learning (CIL): Aims to create intelligent systems that seamlessly assist humans in everyday tasks through continuous interaction and learning [13], [34].

- Human-Agent Symbiosis: Inspired by biological symbiosis, this concept emphasizes the mutual benefits arising from the dynamic interaction between humans and AI systems [35], [36].

- Human-Computer Interaction: Seeks to optimize the design of intuitive and efficient interaction methods between users and autonomous systems [37], [38].

- Collective Hybrid Intelligence: Advocates the notion that true intelligence emerges from the sustained collaboration and co-learning between human and machine counterparts [39].

- Cyber-Physical-Human Systems (CPHSs): Focuses on the seamless integration of digital, physical, and human components to facilitate efficient information exchange and task execution [40].

- Hybrid Intelligence Systems: Examines socio-technical frameworks that synergize human and AI capabilities to address complex problems through collaborative efforts [39], [41], [42].

## III. METHODOLOGY

The methodology employed in this academic paper involved a comprehensive exploration of the current landscape of human-AI collaboration and its related fields, as illustrated in Figure 1. This study followed a systematic review approach, drawing information from primary scholarly research databases, including Google Scholar, Scopus, and IEEE Xplore.

- Artificial Intelligence Generative Content (AIGC): Involves AI systems that generate creative outputs like text, images, or music in a process that promotes collaborative learning and user involvement [19], [20], [43], [44].

- Human-AI team: Suggests that humans and AI function not as separate entities but as integrated parts of a cohesive team, working symbiotically towards shared goals [45], [46].

To gather relevant literature, the databases were queried using a combination of targeted keywords, such as Interactive Machine Learning (IML), Human-Agent Collaboration, Human-Robot Collaboration, Human-Agent Interaction, Human-Robot Interaction, Computer-in-the-Loop,

Each of these research fields embodies unique characteristics and focuses on specific facets of human-technology