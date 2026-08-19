import { useNavigate } from "react-router-dom";
import BackButton from "./backbutton";
import "../styles/style.css";
import { useState } from "react";

function TaskForm() {

    const navigate = useNavigate();

    //----------------------------
    // Task Details
    //----------------------------

    const [taskName, setTaskName] = useState("");
    const [description, setDescription] = useState("");

    //----------------------------
    // Technologies
    //----------------------------

    const [technology, setTechnology] = useState("");
    const [technologies, setTechnologies] = useState([]);
    const [showTechInput, setShowTechInput] = useState(true);

    //----------------------------
    // Tools
    //----------------------------

    const [tool, setTool] = useState("");
    const [tools, setTools] = useState([]);
    const [showToolInput, setShowToolInput] = useState(true);

    //----------------------------
    // Dates
    //----------------------------

    const [startingDate, setStartingDate] = useState("");
    const [deadline, setDeadline] = useState("");

    //----------------------------
    // AI Recommendation
    //----------------------------

    const [complexity, setComplexity] = useState("");
    const [priority, setPriority] = useState("");

    const [estimatedDays, setEstimatedDays] = useState(0);

    const [recommendedDeadline, setRecommendedDeadline] = useState("");

    //--------------------------------------------------
    // AI Schedule Generator
    //--------------------------------------------------

    const calculateSchedule = (
        priority,
        complexity,
        startDate
    ) => {

        if (!priority || !complexity || !startDate)
            return;

        let duration = 5;

        //-----------------------------------
        // Complexity
        //-----------------------------------

        switch (complexity) {

            case "Very Easy":
                duration = 2;
                break;

            case "Easy":
                duration = 5;
                break;

            case "Medium":
                duration = 10;
                break;

            case "Hard":
                duration = 18;
                break;

            case "Expert":
                duration = 30;
                break;

            default:
                duration = 10;
        }

        //-----------------------------------
        // Priority
        //-----------------------------------

        switch (priority) {

            case "Critical":
                duration -= 4;
                break;

            case "High":
                duration -= 2;
                break;

            case "Medium":
                break;

            case "Low":
                duration += 2;
                break;

            case "Very Low":
                duration += 5;
                break;
        }

        if (duration < 1)
            duration = 1;

        const date = new Date(startDate);

        date.setDate(
            date.getDate() + duration
        );

        const deadlineValue =
            date.toISOString().split("T")[0];

        setEstimatedDays(duration);

        setDeadline(deadlineValue);

        setRecommendedDeadline(
            date.toLocaleDateString()
        );
    };

    //--------------------------------------------------
    // Technology
    //--------------------------------------------------

    const handleTechnology = (e) => {

        if (
            e.key === "Enter" &&
            technology.trim() !== ""
        ) {

            e.preventDefault();

            setTechnologies([
                ...technologies,
                technology.trim()
            ]);

            setTechnology("");

            setShowTechInput(false);
        }
    };

    const removeTechnology = (index) => {

        setTechnologies(

            technologies.filter(
                (_, i) => i !== index
            )

        );

    };

    //--------------------------------------------------
    // Tools
    //--------------------------------------------------

    const handleTool = (e) => {

        if (
            e.key === "Enter" &&
            tool.trim() !== ""
        ) {

            e.preventDefault();

            setTools([
                ...tools,
                tool.trim()
            ]);

            setTool("");

            setShowToolInput(false);

        }

    };

    const removeTool = (index) => {

        setTools(

            tools.filter(
                (_, i) => i !== index
            )

        );

    };

    //--------------------------------------------------
    // Submit Task
    //--------------------------------------------------

    const handleConfirm = async () => {

      if (

          !taskName ||

          !description ||

          technologies.length === 0 ||

          tools.length === 0 ||

          !complexity ||

          !priority ||

          !startingDate ||

          !deadline

      ) {

          alert("Please fill all required fields.");

          return;

      }

      const taskData = {

          task_name: taskName,

          description: description,

          technologies: technologies,

          tools_and_ide: tools,

          complexity: complexity,

          priority: priority,

          starting_date: startingDate,

          deadline: deadline,

          duration_days: estimatedDays

      };

      console.log("Task Data :", taskData);

      try {

          const response = await fetch(

              "http://127.0.0.1:8000/task",

              {

                  method: "POST",

                  headers: {

                      "Content-Type": "application/json"

                  },

                  body: JSON.stringify(taskData)

              }

          );

          const result = await response.json();

          if (!response.ok) {

              alert(result.detail || "Unable to create task.");

              return;

          }

          alert("Task Created Successfully");

          setTaskName("");
          setDescription("");

          setTechnology("");
          setTechnologies([]);
          setShowTechInput(true);

          setTool("");
          setTools([]);
          setShowToolInput(true);

          setComplexity("");
          setPriority("");

          setStartingDate("");
          setDeadline("");

          setEstimatedDays(0);

          setRecommendedDeadline("");

          navigate("/admin");

      }

      catch (err) {

          console.log(err);

          alert("Backend Error");

      }

  };

  return (

      <div className="container">

          <BackButton />

          <h1>Task Requirements</h1>

          <label>Task Name</label>
            <input
                type="text"
                placeholder="Enter task name"
                value={taskName}
                onChange={(e)=>setTaskName(e.target.value)}
            />

          <br /><br />

          <label>Description</label>

            <input
            className="task-input"
            type="text"
            placeholder="Describe the task..."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
        />

          <br /><br />

          <label>Technologies</label>

          <br /><br />

          {

              showTechInput &&

              <input

                  type="text"

                  placeholder="Press Enter"

                  value={technology}

                  onChange={(e)=>setTechnology(e.target.value)}

                  onKeyDown={handleTechnology}

              />

          }

          <div>

              {

                  technologies.map((item,index)=>(

                      <span

                          key={index}

                          style={{

                              margin:5,

                              padding:8,

                              border:"1px solid black",

                              display:"inline-block"

                          }}

                      >

                          {item}

                          <button

                              onClick={()=>removeTechnology(index)}

                          >

                              ✖

                          </button>

                      </span>

                  ))

              }

              <button

                  type="button"

                  onClick={()=>setShowTechInput(true)}

              >

                  +

              </button>

          </div>

          <br /><br />

          <label>Tools / IDE</label>

          <br /><br />

          {

              showToolInput &&

              <input

                  type="text"

                  placeholder="Press Enter"

                  value={tool}

                  onChange={(e)=>setTool(e.target.value)}

                  onKeyDown={handleTool}

              />

          }

          <div>

              {

                  tools.map((item,index)=>(

                      <span

                          key={index}

                          style={{

                              margin:5,

                              padding:8,

                              border:"1px solid black",

                              display:"inline-block"

                          }}

                      >

                          {item}

                          <button

                              onClick={()=>removeTool(index)}

                          >

                              ✖

                          </button>

                      </span>

                  ))

              }

              <button

                  type="button"

                  onClick={()=>setShowToolInput(true)}

              >

                  +

              </button>

          </div>

          <br /><br />

          <label>Complexity</label>

          <select

              value={complexity}

              onChange={(e)=>{

                  setComplexity(e.target.value);

                  calculateSchedule(

                      priority,

                      e.target.value,

                      startingDate

                  );

              }}

          >

              <option value="">Select Complexity</option>

              <option value="Very Easy">Very Easy</option>

              <option value="Easy">Easy</option>

              <option value="Medium">Medium</option>

              <option value="Hard">Hard</option>

              <option value="Expert">Expert</option>

          </select>

          <br /><br />

          <label>Priority</label>

          <select

              value={priority}

              onChange={(e)=>{

                  setPriority(e.target.value);

                  calculateSchedule(

                      e.target.value,

                      complexity,

                      startingDate

                  );

              }}

          >

              <option value="">Select Priority</option>

              <option value="Critical">Critical</option>

              <option value="High">High</option>

              <option value="Medium">Medium</option>

              <option value="Low">Low</option>

              <option value="Very Low">Very Low</option>

          </select>

          <br /><br />

          <label>Starting Date</label>

          <input

              type="date"

              value={startingDate}

              onChange={(e)=>{

                  setStartingDate(e.target.value);

                  calculateSchedule(

                      priority,

                      complexity,

                      e.target.value

                  );

              }}

          />

          <br /><br />

          <label>Deadline</label>

                <input
            type="date"
            value={deadline}
            onChange={(e) => setDeadline(e.target.value)}
            min={startingDate}
        />
          <br /><br />

          <button

              onClick={handleConfirm}

          >

              Confirm

          </button>

      </div>
  );
}

export default TaskForm;
