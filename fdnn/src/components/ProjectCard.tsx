import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Typography from "@mui/material/Typography";
import { Link } from "@mui/material";
import { IProjectData } from "./ProjectsPage";

export interface IProjectCardProps {
    project: IProjectData;
}

export default function ProjectCard(props: IProjectCardProps) {
    return (
        <Link underline="none" href={`/projects/${encodeURI(props.project.name)}`}>
            <Card style={{ cursor: "pointer" }} sx={{ width: 500, marginRight: 2, marginBottom: 2 }}>
                <CardContent>
                    <Typography gutterBottom variant="h5" component="div">
                        {props.project.name}
                    </Typography>
                </CardContent>
                <CardContent style={{ paddingBottom: 5, paddingTop: 5 }}>
                    <Typography align="left" variant="button" component="div">
                        Total Images
                    </Typography>
                    <Typography align="left" variant="overline" component="div">
                        {props.project.max_images}
                    </Typography>
                </CardContent>
                <CardContent style={{ paddingBottom: 5, paddingTop: 5 }}>
                    <Typography align="left" variant="button" component="div">
                        Labels
                    </Typography>
                    <Typography align="left" variant="overline" component="div">
                        {props.project.labels.join(",")}
                    </Typography>
                </CardContent>
                <CardContent style={{ paddingBottom: 5, paddingTop: 5 }}>
                    <Typography align="left" variant="button" component="div">
                        Image normalization size
                    </Typography>
                    <Typography align="left" variant="overline" component="div">
                        {props.project.image_height}x{props.project.image_width}
                    </Typography>
                </CardContent>
                <CardContent style={{ paddingBottom: 5, paddingTop: 5 }}>
                    <Typography align="left" variant="button" component="div">
                        Seed
                    </Typography>
                    <Typography align="left" variant="overline" component="div">
                        {props.project.seed}
                    </Typography>
                </CardContent>
            </Card>
        </Link>
    );
}
