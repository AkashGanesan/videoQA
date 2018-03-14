package TestSceneGraph;



import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.Scanner;

import com.google.gson.Gson;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import edu.stanford.nlp.scenegraph.RuleBasedParser;
import edu.stanford.nlp.scenegraph.SceneGraph;

public class TestSceneGraph {



	
	// 1. Read JSON for scenegraphs
	// 2. Create a scene graph
	// 3. Store scene graph
	
	public String text = "A brown fox chases a white rabbit.";
	public static void main(String[] args) throws FileNotFoundException, IOException, ParseException {
		// String jsonPath = args[0];
        Scanner input = new Scanner(System.in);
        RuleBasedParser parser = new RuleBasedParser();
	    SceneGraph sg = parser.parse("");

	    System.out.println("\n#start\n");
        System.out.flush();
	    
	    
	    while (input.hasNextLine()) {
        	sg = parser.parse(input.nextLine());
        	System.out.println(sg.toJSON(0, "0", ""));        	
        	System.out.flush();

	    }
        
	    

	}
}



	
