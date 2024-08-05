import ProjectCard from "./ProjectCard";
import { ProjectCreate } from "./ProjectCreate";
import { IProjectData } from "./ProjectsPage";

interface IProjectGridProps {
    projects: Array<IProjectData>;
    refresh: () => void;
}

export function ProjectGrid(props: IProjectGridProps) {
    return (
        <div
            style={{
                margin: 15,
                textAlign: "left",
            }}
        >
            <ProjectCreate />
            <div
                style={{
                    display: "flex",
                    flexWrap: "wrap",
                    overflow: "hidden",
                    width: "100%",
                    height: "100%",
                    position: "relative",
                    marginTop: 15,
                }}
            >
                {props.projects.map((project) => (
                    <ProjectCard project={project} key={project.name} />
                ))}
            </div>
        </div>
    );
}
