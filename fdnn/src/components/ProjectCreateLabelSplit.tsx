import { Typography } from "@mui/material";
import { ProjectCreateLabelSplitItem } from "./ProjectCreateLabelSplitItem";

interface ProjectCreateLabelSplitProps {
    label_splits: { [key: string]: number };
    setLabel: (key: string, value: number) => void;
}

export function ProjectCreateLabelSplit(props: ProjectCreateLabelSplitProps) {
    return (
        <div style={{ marginTop: 20 }}>
            <Typography variant="h6">Label Split</Typography>
            {Object.keys(props.label_splits).map((key) => (
                <ProjectCreateLabelSplitItem
                    key={key}
                    identifier={key}
                    split={props.label_splits[key]}
                    setLabel={props.setLabel}
                />
            ))}
        </div>
    );
}
