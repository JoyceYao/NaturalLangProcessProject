import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import com.google.common.io.Files;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

/*	Java code for Natural Language Processing project
 * 	Using Stanford CoreNLP
 * 	input: sentences in news articles with/without company relationships
 * 	output: training corpus for Stanford coreNLP with POS and NER tags (for further hand tagging)
 */
public class TrainCorpusCreater{

	static String inputFiles = "relation_final_test.txt";
	static String outputFile = "output.corp";
	static Map<String, String> NERMap;
	static PrintWriter pw;
	static List<CoreMap> mentions;
	static int idx_m = 0;	//index of mentions, reset to 0 when each sentence begins

	public static void main(String[] args) throws IOException {
		
		pw = new PrintWriter(new FileWriter(outputFile));

		prepareNERMap();

		// Setting properties for Stanford CoreNLP pipeline
		Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma, ner, regexner, parse, entitymentions, relation");	
		props.put("regexner.mapping", "jg-regexner.txt");

		StanfordCoreNLP pipeLine = new StanfordCoreNLP(props);

		// inputText will be the text to evaluate in this example
		File inputFile = new File(inputFiles);
		String inputText = Files.toString(inputFile, Charset.forName("UTF-8"));

		Annotation document = new Annotation(inputText);

		// Use the pipeline to annotate the document we created
		pipeLine.annotate(document);

		// Get the annotated sentences
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);

		int idx_s = 0;	//sentence index in output file
		int idx_tp = 0;	//token index per sentence, reset to 0 when each sentence begins
		int idx_ta = 0;	//token index per article

		for (CoreMap sentence : sentences) {
			
			// Get all mentions list in this sentence
			mentions = sentence
					.get(CoreAnnotations.MentionsAnnotation.class);
			
			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				
				// Check if this token is belonged to a mention
				int isMention = checkMentions(pw, idx_s, idx_tp, idx_ta);
				
				//if this token belongs to a mention, reset indexes and continue;
				if(isMention != 0){
					idx_ta++;
					if(isMention == 2){ idx_tp += 1; };
					continue;
				}

				// Extracting Name Entity Recognition
				String ner = token.getString(NamedEntityTagAnnotation.class);
				String nerM = resetNERMap(ner);

				// Extracting Part Of Speech
				String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);

				// Extracting the Lemma
				String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);
				
				// Output to a file 
				pw.println(idx_s + "\t" + nerM + "\t" + idx_tp + "\t"
						+ "O" + "\t" + pos + "\t" + lemma + "\t" + "O" + "\t"
						+ "O" + "\t" + "O");

				idx_tp++;
				idx_ta++;
			}

			// print two empty lines between sentences to fit the default format
			pw.println("");
			pw.println("");
			
			// reset index
			idx_tp = 0;
			idx_m = 0;
			idx_s++;
		}
		
		pw.close();

	}
	
	// Check if a token is belonged to a mention, if it is, write the whole mention to a row, 
	// ex: OneWest/Bank/Group/LLC
	// return: 0: not a mention, 1: part of a mention, 2: end of a mention
	public static int checkMentions(PrintWriter pw, int idx_s, int idx_tp, int idx_ta){

		// If there is no mentions in this sentence, do nothing
		if(mentions == null || mentions.isEmpty()){ return(0); }
		
		// If there is no more mentions left in this sentence, do nothing
		if(mentions.size() == idx_m ){ return(0); }
		
		// Get current mention's start and end position
		CoreMap cMap = mentions.get(idx_m);
		int idx_mb = cMap.get(CoreAnnotations.TokenBeginAnnotation.class);
		int idx_me = cMap.get(CoreAnnotations.TokenEndAnnotation.class);
		
		// If this token is the first element of a mention, output all elements in a row
		if(idx_ta == idx_mb){
			String tags = "", ners = "", texts = "";
			List<CoreLabel> labels = cMap.get(CoreAnnotations.TokensAnnotation.class);
			for(CoreLabel l: labels){
				ners = resetNERMap(l.ner());
				tags += "/" + l.tag();
				texts += "/" + l.originalText();
			}
			pw.println(idx_s + "\t" + ners + "\t" + idx_tp + "\t" + "O" + "\t" + tags.substring(1) + "\t" + texts.substring(1) + "\t" + "O" + "\t" + "O" + "\t" + "O");
			
			// If this mention has only one element, return "end of a mention"
			if(labels.size() == 1){ 
				idx_m++;
				return(2); 
			}else{
				return(1);				
			}
			
		}else if(idx_ta == idx_me-1){
			// this token is the end of a mention
			idx_m++;
			return(2);					
		}else if(idx_ta > idx_mb && idx_ta < idx_me){
			// this token is part of a mention
			return(1);
		}else{
			// this token is not a part of a mention
			return(0);
		}
	}

	public static void prepareNERMap() {
		NERMap = new HashMap<String, String>();
		NERMap.put("LOCATION", "Loc");
		NERMap.put("ORGANIZATION", "Org");
		NERMap.put("PERSON", "Peop");
		NERMap.put("O", "O");
	}
	
	// Reset NER tag to fit the default format
	public static String resetNERMap(String key){
		String nerM = NERMap.get(key);
		if (nerM == null || nerM.isEmpty()) {
			nerM = "OTHER";
		}
		return(nerM);
	}
}

