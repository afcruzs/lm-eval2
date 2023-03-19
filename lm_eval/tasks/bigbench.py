'''
Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models

https://arxiv.org/abs/2206.04615

Language models demonstrate both quantitative improvement and new qualitative capabilities with increasing scale. Despite their potentially transformative impact, these new capabilities are as yet poorly characterized. In order to inform future research, prepare for disruptive new model capabilities, and ameliorate socially harmful effects, it is vital that we understand the present and near-future capabilities and limitations of language models. To address this challenge, we introduce the Beyond the Imitation Game benchmark (BIG-bench). BIG-bench currently consists of 204 tasks, contributed by 442 authors across 132 institutions. Task topics are diverse, drawing problems from linguistics, childhood development, math, common-sense reasoning, biology, physics, social bias, software development, and beyond. BIG-bench focuses on tasks that are believed to be beyond the capabilities of current language models. We evaluate the behavior of OpenAI's GPT models, Google-internal dense transformer architectures, and Switch-style sparse transformers on BIG-bench, across model sizes spanning millions to hundreds of billions of parameters. In addition, a team of human expert raters performed all tasks in order to provide a strong baseline. Findings include: model performance and calibration both improve with scale, but are poor in absolute terms (and when compared with rater performance); performance is remarkably similar across model classes, though with benefits from sparsity; tasks that improve gradually and predictably commonly involve a large knowledge or memorization component, whereas tasks that exhibit "breakthrough" behavior at a critical scale often involve multiple steps or components, or brittle metrics; social bias typically increases with scale in settings with ambiguous context, but this can be improved with prompting.

Homepage: https://github.com/google/BIG-bench
'''
import numpy as np
from lm_eval.api.task import Task
from lm_eval.api.instance import LoglikelihoodInstance
from lm_eval.api.metrics import mean

_CITATION = """
@article{Srivastava2022BeyondTI,
  title={Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models},
  author={Aarohi Srivastava and Abhinav Rastogi and Abhishek Rao and Abu Awal Md Shoeb and Abubakar Abid and Adam Fisch and Adam R. Brown and Adam Santoro and Aditya Gupta and Adri{\`a} Garriga-Alonso and Agnieszka Kluska and Aitor Lewkowycz and Akshat Agarwal and Alethea Power and Alex Ray and Alex Warstadt and Alexander W. Kocurek and Ali Safaya and Ali Tazarv and Alice Xiang and Alicia Parrish and Allen Nie and Aman Hussain and Amanda Askell and Amanda Dsouza and Ameet Annasaheb Rahane and Anantharaman S. Iyer and Anders Andreassen and Andrea Santilli and Andreas Stuhlmuller and Andrew M. Dai and Andrew D. La and Andrew Kyle Lampinen and Andy Zou and Angela Jiang and Angelica Chen and Anh Vuong and Animesh Gupta and Anna Gottardi and Antonio Norelli and Anu Venkatesh and Arash Gholamidavoodi and Arfa Tabassum and Arul Menezes and Arun Kirubarajan and Asher Mullokandov and Ashish Sabharwal and Austin Herrick and Avia Efrat and Aykut Erdem and Ayla Karakacs and Bridget R. Roberts and Bao Sheng Loe and Barret Zoph and Bartlomiej Bojanowski and Batuhan Ozyurt and Behnam Hedayatnia and Behnam Neyshabur and Benjamin Inden and Benno Stein and Berk Ekmekci and Bill Yuchen Lin and Blake Stephen Howald and Cameron Diao and Cameron Dour and Catherine Stinson and Cedrick Argueta and C'esar Ferri Ram'irez and Chandan Singh and Charles Rathkopf and Chenlin Meng and Chitta Baral and Chiyu Wu and Chris Callison-Burch and Chris Waites and Christian Voigt and Christopher D. Manning and Christopher Potts and Cindy Tatiana Ramirez and Clara Rivera and Clemencia Siro and Colin Raffel and Courtney Ashcraft and Cristina Garbacea and Damien Sileo and Daniel H Garrette and Dan Hendrycks and Dan Kilman and Dan Roth and Daniel Freeman and Daniel Khashabi and Daniel Levy and Daniel Gonz'alez and Danny Hernandez and Danqi Chen and Daphne Ippolito and Dar Gilboa and David Dohan and D. Drakard and David Jurgens and Debajyoti Datta and Deep Ganguli and Denis Emelin and Denis Kleyko and Deniz Yuret and Derek Chen and Derek Tam and Dieuwke Hupkes and Diganta Misra and Dilyar Buzan and Dimitri Coelho Mollo and Diyi Yang and Dong-Ho Lee and Ekaterina Shutova and Ekin Dogus Cubuk and Elad Segal and Eleanor Hagerman and Elizabeth Barnes and Elizabeth P. Donoway and Ellie Pavlick and Emanuele Rodol{\`a} and Emma FC Lam and Eric Chu and Eric Tang and Erkut Erdem and Ernie Chang and Ethan A. Chi and Ethan Dyer and Ethan J. Jerzak and Ethan Kim and Eunice Engefu Manyasi and Evgenii Zheltonozhskii and Fan Xia and Fatemeh Siar and Fernando Mart'inez-Plumed and Francesca Happ'e and François Chollet and Frieda Rong and Gaurav Mishra and Genta Indra Winata and Gerard de Melo and Germ{\'a}n Kruszewski and Giambattista Parascandolo and Giorgio Mariani and Gloria Wang and Gonzalo Jaimovitch-L'opez and Gregor Betz and Guy Gur-Ari and Hana Galijasevic and Han Sol Kim and Hannah Rashkin and Hanna Hajishirzi and Harsh Mehta and Hayden Bogar and Henry Shevlin and Hinrich Sch{\"u}tze and Hiromu Yakura and Hongming Zhang and Hubert Wong and Ian Aik-Soon Ng and Isaac Noble and Jaap Jumelet and Jack Geissinger and John Kernion and Jacob Hilton and Jaehoon Lee and Jaime Fern{\'a}ndez Fisac and J. Brooker Simon and James Koppel and James Zheng and James Zou and Jan Koco'n and Jana Thompson and Jared Kaplan and Jarema Radom and Jascha Narain Sohl-Dickstein and Jason Phang and Jason Wei and Jason Yosinski and Jekaterina Novikova and Jelle Bosscher and Jenni Marsh and Jeremy Kim and Jeroen Taal and Jesse Engel and Jesujoba Oluwadara Alabi and Jiacheng Xu and Jiaming Song and Jillian Tang and Jane W Waweru and John Burden and John Miller and John U. Balis and Jonathan Berant and Jorg Frohberg and Jos Rozen and Jos{\'e} Hern{\'a}ndez-Orallo and Joseph Boudeman and Joseph Jones and Joshua B. Tenenbaum and Joshua S. Rule and Joyce Chua and Kamil Kanclerz and Karen Livescu and Karl Krauth and Karthik Gopalakrishnan and Katerina Ignatyeva and Katja Markert and Kaustubh D. Dhole and Kevin Gimpel and Kevin Ochieng’ Omondi and Kory Wallace Mathewson and Kristen Chiafullo and Ksenia Shkaruta and Kumar Shridhar and Kyle McDonell and Kyle Richardson and Laria Reynolds and Leo Gao and Li Zhang and Liam Dugan and Lianhui Qin and Lidia Contreras-Ochando and Louis-Philippe Morency and Luca Moschella and Luca Lam and Lucy Noble and Ludwig Schmidt and Luheng He and Luis Oliveros Col'on and Luke Metz and Lutfi Kerem cSenel and Maarten Bosma and Maarten Sap and Maartje ter Hoeve and Madotto Andrea and Maheen Saleem Farooqi and Manaal Faruqui and Mantas Mazeika and Marco Baturan and Marco Marelli and Marco Maru and M Quintana and Marie Tolkiehn and Mario Giulianelli and Martha Lewis and Martin Potthast and Matthew Leavitt and Matthias Hagen and M'aty'as Schubert and Medina Baitemirova and Melissa Arnaud and Melvin Andrew McElrath and Michael A. Yee and Michael Cohen and Mi Gu and Michael I. Ivanitskiy and Michael Starritt and Michael Strube and Michal Swkedrowski and Michele Bevilacqua and Michihiro Yasunaga and Mihir Kale and Mike Cain and Mimee Xu and Mirac Suzgun and Monica Tiwari and Mohit Bansal and Moin Aminnaseri and Mor Geva and Mozhdeh Gheini and T MukundVarma and Nanyun Peng and Nathan Chi and Nayeon Lee and Neta Gur-Ari Krakover and Nicholas Cameron and Nicholas S. Roberts and Nicholas Doiron and Nikita Nangia and Niklas Deckers and Niklas Muennighoff and Nitish Shirish Keskar and Niveditha Iyer and Noah Constant and Noah Fiedel and Nuan Wen and Oliver Zhang and Omar Agha and Omar Elbaghdadi and Omer Levy and Owain Evans and Pablo Antonio Moreno Casares and Parth Doshi and Pascale Fung and Paul Pu Liang and Paul Vicol and Pegah Alipoormolabashi and Peiyuan Liao and Percy Liang and Peter W. Chang and Peter Eckersley and Phu Mon Htut and Pi-Bei Hwang and P. Milkowski and Piyush S. Patil and Pouya Pezeshkpour and Priti Oli and Qiaozhu Mei and QING LYU and Qinlang Chen and Rabin Banjade and Rachel Etta Rudolph and Raefer Gabriel and Rahel Habacker and Ram'on Risco Delgado and Rapha{\"e}l Milli{\`e}re and Rhythm Garg and Richard Barnes and Rif A. Saurous and Riku Arakawa and Robbe Raymaekers and Robert Frank and Rohan Sikand and Roman Novak and Roman Sitelew and Ronan Le Bras and Rosanne Liu and Rowan Jacobs and Rui Zhang and Ruslan Salakhutdinov and Ryan Chi and Ryan Lee and Ryan Stovall and Ryan Teehan and Rylan Yang and Sahib J. Singh and Saif M. Mohammad and Sajant Anand and Sam Dillavou and Sam Shleifer and Sam Wiseman and Samuel Gruetter and Sam Bowman and Samuel S. Schoenholz and Sanghyun Han and Sanjeev Kwatra and Sarah A. Rous and Sarik Ghazarian and Sayan Ghosh and Sean Casey and Sebastian Bischoff and Sebastian Gehrmann and Sebastian Schuster and Sepideh Sadeghi and Shadi S. Hamdan and Sharon Zhou and Shashank Srivastava and Sherry Shi and Shikhar Singh and Shima Asaadi and Shixiang Shane Gu and Shubh Pachchigar and Shubham Toshniwal and Shyam Upadhyay and Shyamolima Debnath and Siamak Shakeri and Simon Thormeyer and Simone Melzi and Siva Reddy and Sneha Priscilla Makini and Soo-hwan Lee and Spencer Bradley Torene and Sriharsha Hatwar and Stanislas Dehaene and Stefan Divic and Stefano Ermon and Stella Rose Biderman and Stephanie C. Lin and S. Prasad and Steven T. Piantadosi and Stuart M. Shieber and Summer Misherghi and Svetlana Kiritchenko and Swaroop Mishra and Tal Linzen and Tal Schuster and Tao Li and Tao Yu and Tariq A. Ali and Tatsuo Hashimoto and Te-Lin Wu and Theo Desbordes and Theodore Rothschild and Thomas Phan and Tianle Wang and Tiberius Nkinyili and Timo Schick and T. N. Kornev and Timothy Telleen-Lawton and Titus Tunduny and Tobias Gerstenberg and Trenton Chang and Trishala Neeraj and Tushar Khot and Tyler O’Brien Shultz and Uri Shaham and Vedant Misra and Vera Demberg and Victoria Nyamai and Vikas Raunak and Vinay Venkatesh Ramasesh and Vinay Uday Prabhu and Vishakh Padmakumar and Vivek Srikumar and William Fedus and William Saunders and William Zhang and W Vossen and Xiang Ren and Xiaoyu Tong and Xinyi Wu and Xudong Shen and Yadollah Yaghoobzadeh and Yair Lakretz and Yang Song and Yasaman Bahri and Ye Ji Choi and Yichi Yang and Yiding Hao and Yifu Chen and Yonatan Belinkov and Yu Hou and Yu Hou and Yuntao Bai and Zachary Seid and Zhao Xinran and Zhuoye Zhao and Zi Fu Wang and Zijie J. Wang and Zirui Wang and Ziyi Wu and Sahib Singh and Uri Shaham},
  journal={ArXiv},
  year={2022},
  volume={abs/2206.04615}
}
"""


'''
TODO: this is largely just a RFC/Proposal.
- This relies on HF's port to fully respect the task / prompt format. TODO: verify this? talk to HF? - IS not clear where their preprocessing code is.
- List all tasks per type (loglikelihood is just one task)
- Possibly decompose each task type on a metric basis (e.g. if departs from simple accuraccy)

'''

class BigBenchBaseLoglikelihood(Task):
    VERSION = None

    OUTPUT_TYPE = "loglikelihood"

    DATASET_PATH = 'bigbench'

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        raise ValueError("No test docs, just validation docs")
        
    def has_validation_docs(self):
        return True
    
    def has_test_docs(self):
        return False
    
    def has_training_docs(self):
        return True

    def doc_to_text(self, doc):
        return doc['inputs']

    def should_decontaminate(self):
        # TODO: do not really get what this is supposed to be
        return False

    def doc_to_target(self, doc):
        assert len(doc['targets']) == 1
        return " {}".format(doc['targets'][0])

    def construct_requests(self, doc, ctx, **kwargs):
        return [
            LoglikelihoodInstance(
                doc=doc,
                arguments=(ctx, " {}".format(target)),
                id_=idx,
                **kwargs
            )
            for idx, target in enumerate(doc['multiple_choice_targets'])
        ]

    def process_results(self, doc, results):
        results = [res[0] for res in results] # only retain loglikelihoods, discard is_greedy TODO: do we need is_greedy anywhere? 
        assert len(doc["targets"]) == 1
        gold = doc["targets"][0]

        acc = 1.0 if np.argmax(results) == gold else 0.0

        # TODO: does it make sense to normalize by char length? (as MultipleChoiceTask does)
        completion_len = np.array([float(len(choice)) for choice in doc["multiple_choice_targets"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm,
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
        }

class Snarks(BigBenchBaseLoglikelihood):
    DATASET_NAME = 'snarks'