/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import edu.stanford.nlp.scenegraph.RuleBasedParser;
import edu.stanford.nlp.scenegraph.SceneGraph;


/**
 *
 * @author akash
 */
public class SceneGraphTest {
    

    public static void main(String[] args){
        String sentence = "A brown fox chases a white rabbit.";

        RuleBasedParser parser = new RuleBasedParser();
        SceneGraph sg = parser.parse(sentence);

        //printing the scene graph in a readable format
        System.out.println(sg.toReadableString()); 

        //printing the scene graph in JSON form
        //System.out.println(sg.toJSON()); 
        }
}